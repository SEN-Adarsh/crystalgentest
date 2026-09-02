import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Element, Structure
from scipy.spatial import Voronoi

from mattergen.diffusion.redox import ANION_NUMBERS, FIXED_OXIDATION, REDOX_WINDOW

# Element tables are derived from the canonical atomic-number-keyed tables in
# mattergen.diffusion.redox so the placer and the training loss agree on which
# metals are redox active and what nominal charges the spectators carry.
REDOX_LIMITS: Dict[str, Tuple[float, float]] = {
    Element.from_Z(z).symbol: window for z, window in REDOX_WINDOW.items()
}
OXIDATION_MAP: Dict[str, float] = {
    Element.from_Z(z).symbol: ox for z, ox in FIXED_OXIDATION.items()
}
# Redox-active metals sit at the low end of their window in the discharged state.
OXIDATION_MAP.update({symbol: window[0] for symbol, window in REDOX_LIMITS.items()})

ANIONS: Tuple[str, ...] = tuple(Element.from_Z(z).symbol for z in ANION_NUMBERS)

# An anion-anion bond means the pair is oxidised: peroxide O2(2-) instead of two
# O(2-), persulfide S2(2-) instead of two S(2-). Either way the pair carries 2
# fewer electrons than the isolated ions, so each dimer found reduces the total
# anion charge by 2. Cutoffs are safely above the free-molecule bond length and
# well below the shortest non-bonded anion-anion contact in an oxide or sulfide.
#
# ponytail: homonuclear pairs only. Mixed dimers (O-F, S-Cl) are vanishingly rare
# in cathode hosts; add them here if a generated structure ever needs it.
ANION_DIMER_CUTOFF: Dict[str, float] = {
    "O": 1.60,  # peroxide O-O is 1.49
    "F": 1.60,  # F2 is 1.41
    "S": 2.30,  # persulfide S-S is 2.05
    "Cl": 2.20,  # Cl2 is 1.99
}


class PhysicsInformedLiPlacer:
    def __init__(
        self,
        min_li_dist: float = 2.30,
        # Li-to-framework-cation floor. Two cations repel, so real Li-TM contacts in
        # oxides are 2.5-2.9 A (face- to edge-sharing octahedra); 2.40 is a permissive
        # floor below that. The old 1.85 was a Li-anion number applied to cations and
        # let Li land 1.86-1.90 A from Ni/Co in sampled hosts.
        min_tm_dist: float = 2.40,
        min_anion_dist: float = 1.70,
        max_anion_dist: float = 2.70,
        target_coordination: Optional[str] = None,
    ):
        self.min_li_dist = min_li_dist
        self.min_tm_dist = min_tm_dist
        self.min_anion_dist = min_anion_dist
        self.max_anion_dist = max_anion_dist
        self.target_coordination = target_coordination

    def count_anion_dimers(self, structure: Structure) -> int:
        """Number of anion-anion bonded pairs in the cell.

        Each site is matched at most once, shortest bonds first, so a linear S3
        chain counts as one dimer plus one lone anion rather than two dimers.
        """
        anion_ix = [i for i, s in enumerate(structure) if s.specie.symbol in ANIONS]

        bonds: List[Tuple[float, int, int]] = []
        for a_pos, i in enumerate(anion_ix):
            symbol = structure[i].specie.symbol
            cutoff = ANION_DIMER_CUTOFF.get(symbol)
            if cutoff is None:
                continue
            for j in anion_ix[a_pos + 1 :]:
                if structure[j].specie.symbol != symbol:
                    continue
                d = structure.lattice.get_distance_and_image(
                    structure[i].frac_coords, structure[j].frac_coords
                )[0]
                if d < cutoff:
                    bonds.append((d, i, j))

        matched: set[int] = set()
        dimers = 0
        for _, i, j in sorted(bonds):
            if i in matched or j in matched:
                continue
            matched.update((i, j))
            dimers += 1

        return dimers

    def calculate_redox_capacity(self, structure: Structure) -> int:
        """Number of Li the host can accept without leaving the accessible redox window.

        Charge neutrality of the lithiated cell requires

            n_Li + (spectator cation charge) + sum_i x_i = |anion charge|

        where x_i is the oxidation state of redox-active metal i. Driving every
        metal to the bottom of its window maximises n_Li, and the result is
        additionally capped by the total electrons the metals can accept.

        Returns 0 for hosts with no redox-active metal: those cannot cycle Li and
        are rejected rather than stuffed with Li that has nowhere to go. Also
        returns 0 for hosts containing an element absent from the oxidation tables,
        since guessing its charge produces a capacity that looks authoritative and
        is not, and 0 for hosts that cannot be charge balanced at all (see below).
        """
        anion_charge = 0.0
        spectator_charge = 0.0
        sum_lower = 0.0
        sum_upper = 0.0
        num_redox_metals = 0

        for site in structure:
            symbol = site.specie.symbol
            if symbol in ANIONS:
                anion_charge += abs(OXIDATION_MAP[symbol])
            elif symbol in REDOX_LIMITS:
                lower, upper = REDOX_LIMITS[symbol]
                sum_lower += lower
                sum_upper += upper
                num_redox_metals += 1
            elif symbol in OXIDATION_MAP:
                spectator_charge += OXIDATION_MAP[symbol]
            else:
                # Unknown charge: no capacity can be trusted for this host.
                return 0

        if num_redox_metals == 0:
            return 0

        # Anion-anion bonds have already consumed some of the charge the metals
        # would otherwise have to balance.
        anion_charge -= 2.0 * self.count_anion_dimers(structure)

        # Charge the redox metals must carry for the delithiated host to be neutral.
        needed = anion_charge - spectator_charge

        # Feasibility: if the metals cannot reach that charge even fully oxidised,
        # the host does not exist as written. Sampled frameworks are routinely
        # oxygen-rich this way (MnO4 wants Mn(8+), NiSnO8 wants Ni(12+)), and
        # without this check the capacity below silently clamps to max_electrons
        # and reports a confident number for an impossible structure.
        if needed > sum_upper:
            return 0

        capacity = needed - sum_lower
        capacity = min(capacity, sum_upper - sum_lower)

        return int(np.floor(max(capacity, 0.0)))

    def generate_periodic_voronoi_sites(self, structure: Structure) -> List[np.ndarray]:
        cart_coords = structure.cart_coords
        lattice_mat = structure.lattice.matrix
        all_points = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shift = dx * lattice_mat[0] + dy * lattice_mat[1] + dz * lattice_mat[2]
                    all_points.append(cart_coords + shift)

        all_points = np.vstack(all_points)
        vor = Voronoi(all_points)

        inv_lat = structure.lattice.inv_matrix
        candidate_fracs = []

        for vertex in vor.vertices:
            frac = np.dot(vertex, inv_lat)
            if np.all(frac >= -0.02) and np.all(frac < 1.02):
                candidate_fracs.append(frac % 1.0)

        # Merge close vertices within 0.35 Å
        unique_fracs = []
        for f in candidate_fracs:
            if not any(structure.lattice.get_distance_and_image(f, uf)[0] < 0.35 for uf in unique_fracs):
                unique_fracs.append(f)

        return unique_fracs

    def classify_coordination(self, structure: Structure, frac_coord: np.ndarray) -> Tuple[int, str]:
        anion_dists = []
        for site in structure:
            if site.specie.symbol in ANIONS:
                d = structure.lattice.get_distance_and_image(frac_coord, site.frac_coords)[0]
                if self.min_anion_dist <= d <= self.max_anion_dist:
                    anion_dists.append(d)

        cn = len(anion_dists)
        if cn == 6:
            geom = "octahedral"
        elif cn == 4:
            geom = "tetrahedral"
        else:
            geom = f"distorted_{cn}"

        return cn, geom

    def compute_site_potentials(
        self, structure: Structure, candidate_fracs: List[np.ndarray]
    ) -> List[float]:
        """Electrostatic potential felt by a unit positive probe at each candidate site.

        A minimum-image Coulomb sum over the framework ions, using nominal
        oxidation states. Lower (more negative) is a deeper well for Li+.

        ponytail: real-space minimum-image sum, not Ewald. The candidates are
        ranked against each other inside one cell, so the omitted long-range tail
        is nearly constant across them and does not change the ordering. Swap in
        pymatgen's EwaldSummation if absolute site energies are ever needed.
        """
        potentials: List[float] = []
        charges = [OXIDATION_MAP.get(site.specie.symbol, 2.0) for site in structure]

        for frac in candidate_fracs:
            potential = 0.0
            for charge, site in zip(charges, structure):
                d = structure.lattice.get_distance_and_image(frac, site.frac_coords)[0]
                if d > 0.1:
                    potential += charge / d
            potentials.append(potential)

        return potentials

    def place_lithium(self, structure: Structure) -> Optional[Structure]:
        target_li = self.calculate_redox_capacity(structure)
        if target_li == 0:
            # No redox-active metal, or no charge headroom: not a cathode host.
            return None

        candidates = self.generate_periodic_voronoi_sites(structure)

        if not candidates:
            return None

        # 1. Geometric Filtering
        valid_candidates = []
        for frac in candidates:
            tm_dists = [
                structure.lattice.get_distance_and_image(frac, s.frac_coords)[0]
                for s in structure if s.specie.symbol not in ANIONS
            ]
            min_tm_d = min(tm_dists) if tm_dists else 999.0

            anion_dists = [
                structure.lattice.get_distance_and_image(frac, s.frac_coords)[0]
                for s in structure if s.specie.symbol in ANIONS
            ]
            min_anion_d = min(anion_dists) if anion_dists else 0.0

            if min_tm_d >= self.min_tm_dist and min_anion_d >= self.min_anion_dist:
                cn, geom = self.classify_coordination(structure, frac)
                if self.target_coordination is None or geom == self.target_coordination:
                    valid_candidates.append((frac, cn, geom))

        # Adaptive fallback: relax constraints slightly if strict cutoffs found no voids
        if not valid_candidates:
            for frac in candidates:
                tm_dists = [
                    structure.lattice.get_distance_and_image(frac, s.frac_coords)[0]
                    for s in structure if s.specie.symbol not in ANIONS
                ]
                min_tm_d = min(tm_dists) if tm_dists else 999.0

                if min_tm_d >= self.min_tm_dist - 0.30:
                    cn, geom = self.classify_coordination(structure, frac)
                    valid_candidates.append((frac, cn, geom))

        if not valid_candidates:
            return None

        candidate_fracs = [item[0] for item in valid_candidates]

        # 2. Electrostatic Potential Well Ranking
        potentials = self.compute_site_potentials(structure, candidate_fracs)
        
        ranked_sites = sorted(
            zip(candidate_fracs, potentials, [item[2] for item in valid_candidates]),
            key=lambda x: x[1]
        )

        # 3. Greedy Sublattice Placement with Li-Li Repulsion
        final_struct = structure.copy()
        placed_li_coords = []

        for frac, pot, geom in ranked_sites:
            if len(placed_li_coords) >= target_li:
                break

            clash = False
            for placed in placed_li_coords:
                d = final_struct.lattice.get_distance_and_image(frac, placed)[0]
                if d < self.min_li_dist:
                    clash = True
                    break

            if not clash:
                placed_li_coords.append(frac)
                final_struct.append("Li", frac)

        if len(placed_li_coords) == 0:
            return None

        return final_struct


def process_directory(input_dir: Path, output_dir: Path, coordination: Optional[str] = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_files = sorted(list(input_dir.glob("*.cif")))
    print(f"Loaded {len(cif_files)} scaffold CIFs from: {input_dir}")

    placer = PhysicsInformedLiPlacer(target_coordination=coordination)

    success_count = 0
    for cif in cif_files:
        try:
            struct = Structure.from_file(str(cif))
            lithiated = placer.place_lithium(struct)
            
            if lithiated is not None:
                success_count += 1
                formula = lithiated.composition.reduced_formula
                out_path = output_dir / f"lithiated_{cif.stem}_{formula}.cif"
                lithiated.to(filename=str(out_path))

                n_tm = len([s for s in lithiated if s.specie.symbol in REDOX_LIMITS])
                n_li = len([s for s in lithiated if s.specie.symbol == "Li"])
                print(
                    f"[SUCCESS] {cif.name} -> {formula} | "
                    f"Li/TM: {n_li / max(1, n_tm):.2f} | Total Atoms: {len(lithiated)}"
                )
            else:
                print(
                    f"[REJECTED] {cif.name}: no redox-active metal, an element with no "
                    "tabulated oxidation state, a charge that no accessible oxidation "
                    "state can balance, or no interstitial site matched the steric "
                    "criteria."
                )
        except Exception as e:
            print(f"[ERROR] Failed processing {cif.name}: {e}")

    print(f"\nProcessing Complete: {success_count}/{len(cif_files)} structures successfully lithiated.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-Informed Deterministic Li+ Placer Engine")
    parser.add_argument("--input_dir", type=str, default="scaffolds", help="Path to input scaffolds directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to output lithiated CIF directory")
    parser.add_argument("--coord", type=str, default=None, choices=["octahedral", "tetrahedral"], help="Force specific coordination geometry")
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output_dir), args.coord)