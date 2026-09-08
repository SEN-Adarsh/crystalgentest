import argparse
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure

# MP API notes (verified live, Sep 2026):
# - /materials/thermo serves up to three docs per material, tagged thermo_type
#   in {"GGA_GGA+U", "GGA_GGA+U_R2SCAN", "r2SCAN"}. Only pure GGA_GGA+U docs
#   are on the classic MP2020 footing: r2SCAN energies sit 1-3 eV/atom off,
#   and GGA_GGA+U_R2SCAN docs carry r2SCAN-scale energies for elemental
#   metals (e.g. Co: -13.2 vs -7.1), which silently poisons every mixture
#   containing that element.
# - /materials/summary picks one doc per material without keeping the
#   footing consistent across a chemical system, so it cannot build a hull.
# - No endpoint exposes uncorrected energies, so the repo's
#   TRI110Compatibility2024 (which needs them on both sides) cannot be used
#   against the live API.
# Candidate treatment: CHGNet energies are already on the MP-corrected scale
# (validated on 10 real MP structures, median offset +76 meV/atom, Sep
# 2026) and must NOT be pushed through MaterialsProject2020Compatibility
# (that double-corrects by ~-0.67 eV/atom). References below and candidates
# therefore share one footing with no correction applied on either side.


class DirectMPRester:
    """
    Direct REST client for the Materials Project API to fetch a complete closed
    chemical system including elemental endpoints, binaries, and ternaries.

    Uses /materials/thermo filtered to pure GGA_GGA+U docs (the classic
    MP2020 footing); r2SCAN docs and the /materials/summary endpoint mix
    energy footings and must not be used to build a hull.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.materialsproject.org"
        self.headers = {"X-API-KEY": self.api_key, "accept": "application/json"}
        self._chemsys_cache: Dict[str, List[dict]] = {}

    def _fetch_chemsys(self, chemsys: str) -> List[dict]:
        """One chemsys, GGA/GGA+U thermo docs, paginated. Cached per chemsys."""
        if chemsys in self._chemsys_cache:
            return self._chemsys_cache[chemsys]

        url = f"{self.base_url}/materials/thermo/"
        items: List[dict] = []
        # ponytail: pagination by _skip; a 400 on _skip (endpoint without it)
        # degrades to the first page with a warning rather than crashing.
        skip = 0
        while True:
            params = {
                "chemsys": chemsys,
                "_fields": "material_id,formula_pretty,composition,energy_per_atom,thermo_type",
                "_limit": 1000,
                "_skip": skip,
            }
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            if response.status_code != 200:
                if skip > 0:
                    print(f"  [WARN] Pagination failed for {chemsys}: {response.text[:120]}")
                    break
                raise RuntimeError(f"MP API error ({response.status_code}): {response.text}")
            raw = response.json().get("data", [])
            items.extend(d for d in raw if d.get("thermo_type") == "GGA_GGA+U")
            if len(raw) < 1000:
                break
            skip += 1000
            print(f"  [MP API] {chemsys}: paginating past {skip} entries")

        self._chemsys_cache[chemsys] = items
        return items

    def get_phase_diagram_entries(self, elements: List[str]) -> List[PDEntry]:
        """
        Queries every subsystem of the target elements separately (the API has
        no single closed-subsystem query) and returns one PDEntry per material.
        Served energies are already MP2020-corrected, so they are used as
        served.
        """
        elem_set = set(elements)
        entries: List[PDEntry] = []
        seen: set = set()

        for size in range(1, len(elements) + 1):
            for subset in combinations(sorted(elem_set), size):
                for item in self._fetch_chemsys("-".join(subset)):
                    mid = item.get("material_id")
                    if mid in seen:
                        continue
                    comp_dict = item.get("composition", {})
                    if not set(comp_dict.keys()).issubset(elem_set):
                        continue
                    energy_per_atom = item.get("energy_per_atom")
                    total_atoms = sum(comp_dict.values())
                    if energy_per_atom is None or total_atoms == 0:
                        continue
                    seen.add(mid)
                    entries.append(
                        PDEntry(
                            composition=Composition(comp_dict),
                            energy=energy_per_atom * total_atoms,
                            name=item.get("formula_pretty", mid),
                            attribute=mid,
                        )
                    )
        return entries


class HullStabilityEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MP_API_KEY")
        if not self.api_key:
            raise ValueError("Materials Project API key is required.")
        self.client = DirectMPRester(self.api_key)
        self._pd_cache: Dict[str, Optional[PhaseDiagram]] = {}

    def get_phase_diagram(self, elements: List[str]) -> Optional[PhaseDiagram]:
        canonical_chemsys = "-".join(sorted(elements))

        if canonical_chemsys not in self._pd_cache:
            print(f"  [MP API] Querying closed phase diagram for system: {canonical_chemsys}...")
            try:
                entries = self.client.get_phase_diagram_entries(elements)
                if not entries:
                    print(f"  [WARN] No database entries found for {canonical_chemsys}")
                    self._pd_cache[canonical_chemsys] = None
                    return None

                pd_obj = PhaseDiagram(entries)
                self._pd_cache[canonical_chemsys] = pd_obj
            except Exception as e:
                print(f"  [WARN] Phase diagram construction failed for {canonical_chemsys}: {e}")
                self._pd_cache[canonical_chemsys] = None
                return None

        return self._pd_cache[canonical_chemsys]

    def evaluate_candidate(self, struct: Structure, energy_per_atom: float) -> Optional[Dict]:
        composition = struct.composition
        elements = [el.symbol for el in composition.elements]

        pd_ref = self.get_phase_diagram(elements)
        if pd_ref is None:
            return None

        # CHGNet energies share the MP-corrected footing of the references
        # (see module notes); no correction is applied on either side.
        candidate_entry = PDEntry(
            composition=composition,
            energy=energy_per_atom * len(struct),
            name=composition.reduced_formula,
        )

        try:
            e_above_hull_ev = pd_ref.get_e_above_hull(candidate_entry)
            e_above_hull_mev = e_above_hull_ev * 1000.0

            try:
                decomp_phases = pd_ref.get_decomposition(composition)
                decomp_str = " + ".join(
                    [f"{frac:.2f} {entry.name}" for entry, frac in decomp_phases.items()]
                )
            except Exception:
                decomp_str = "N/A"

            if e_above_hull_mev <= 0.0:
                stability_tier = "Thermodynamically Stable (On Hull)"
            elif e_above_hull_mev <= 50.0:
                stability_tier = "Synthetically Accessible (<50 meV)"
            elif e_above_hull_mev <= 100.0:
                stability_tier = "Metastable (50-100 meV)"
            else:
                stability_tier = "Unstable (>100 meV)"

            return {
                "formula": composition.reduced_formula,
                "e_above_hull_mev_atom": round(e_above_hull_mev, 2),
                "stability_tier": stability_tier,
                "decomposition_pathway": decomp_str,
            }
        except Exception as e:
            print(f"  [WARN] Hull evaluation error for {composition.reduced_formula}: {e}")
            return None


def process_screening_results(
    csv_path: Path, cif_dir: Path, output_csv: Path, api_key: Optional[str] = None
):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "Converged"].copy()

    evaluator = HullStabilityEvaluator(api_key=api_key)
    hull_results = []

    print(f"\nEvaluating convex hull stability for {len(df)} relaxed candidates...\n")

    for _, row in df.iterrows():
        cif_name = row["file"]
        relaxed_cif = cif_dir / f"relaxed_{cif_name}"

        if not relaxed_cif.exists():
            relaxed_cif = cif_dir / cif_name

        if not relaxed_cif.exists():
            continue

        try:
            struct = Structure.from_file(str(relaxed_cif))
            energy_per_atom = float(row["energy_per_atom_eV"])

            metrics = evaluator.evaluate_candidate(struct, energy_per_atom)

            if metrics is not None:
                combined = {
                    "file": cif_name,
                    "formula": metrics["formula"],
                    "total_atoms": len(struct),
                    "energy_per_atom_eV": energy_per_atom,
                    "vol_change_pct": row.get("vol_change_pct", np.nan),
                    "capacity_mAh_g": row.get("theoretical_capacity_mAh_g", 0.0),
                    "e_above_hull_mev_atom": metrics["e_above_hull_mev_atom"],
                    "stability_tier": metrics["stability_tier"],
                    "decomposition_phases": metrics["decomposition_pathway"],
                }
                hull_results.append(combined)

                print(
                    f"[SCREENED] {metrics['formula']:<16} | "
                    f"E_hull: {metrics['e_above_hull_mev_atom']:>7.1f} meV/atom | "
                    f"Tier: {metrics['stability_tier']}"
                )
        except Exception as e:
            print(f"[ERROR] Failed evaluating {cif_name}: {e}")

    if hull_results:
        result_df = pd.DataFrame(hull_results)
        result_df = result_df.sort_values(by="e_above_hull_mev_atom", ascending=True)
        result_df.to_csv(output_csv, index=False)
        print(f"\nScreening finished! Successfully evaluated {len(hull_results)} candidates.")
        print(f"Results written to:\n{output_csv}")
    else:
        print("\nNo candidates could be evaluated against the MP Phase Diagram.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materials Project Convex Hull Screening")
    parser.add_argument("--csv_path", type=str, default="screened_results/screening_results.csv")
    parser.add_argument("--cif_dir", type=str, default="screened_results")
    parser.add_argument(
        "--output_csv", type=str, default="screened_results/convex_hull_screened.csv"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Materials Project API key. Falls back to the MP_API_KEY environment variable.",
    )
    args = parser.parse_args()

    process_screening_results(
        Path(args.csv_path), Path(args.cif_dir), Path(args.output_csv), args.api_key
    )
