"""Cycling screen: does the framework survive delithiation at every stage?

Feature 4. A cathode is not judged on its two endpoints. If a material holds 3 Li,
it has to stay intact at Li3, Li2, Li1 and Li0 - frameworks typically collapse at
an intermediate composition, not at full discharge. So this script builds the whole
delithiation ladder from each generated pair, relaxes every rung with CHGNet, and
reports per rung:

  step voltage    Plateau between this rung and the previous, referenced to Li
                  metal. Together the rungs are the voltage curve.
  step dV         Volume jump across this single Li removal. A sharp step cracks
                  particles even when the end-to-end swing looks acceptable.
  total dV        Volume relative to the fully lithiated state. Real cathodes sit
                  low (LiCoO2 ~2%, LiFePO4 ~7%).
  topotactic      Is the framework at this rung still the framework we designed,
                  or did the network re-bond into something else? This is the
                  "doesn't break/decompose" criterion, checked at every stage.
  Li clearance    Tightest gap between a remaining Li and the framework, as a proxy
                  for the diffusion bottleneck.

A candidate passes only if every rung is topotactic and both volume budgets and the
voltage window hold across the whole curve.

Usage:
    python -m mattergen.cycling_screen --pairs_dir results/pairs
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

# Screening thresholds. ponytail: literature rules of thumb, not fitted. Widen
# with --max_volume_change etc. rather than editing them here.
MAX_VOLUME_CHANGE_PCT = 10.0
# A single sharp step cracks particles even when the end-to-end swing looks fine,
# so the per-step budget is tighter than the total.
MAX_STEP_VOLUME_CHANGE_PCT = 6.0
MIN_VOLTAGE = 2.0
MAX_VOLTAGE = 5.0

# bcc Li metal, the standard anode reference for cathode voltages.
LI_METAL_LATTICE = 3.44


def average_voltage(
    energy_lithiated: float, energy_host: float, num_li: int, energy_li_metal: float
) -> float:
    """Average intercalation voltage in volts, referenced to Li metal.

        V = -(E_lithiated - E_host - n * E_Li) / n

    Lithiation is exothermic, so the bracket is negative and V comes out positive.
    Energies are totals in eV; `energy_li_metal` is per atom.
    """
    if num_li <= 0:
        raise ValueError("num_li must be positive to define a voltage")
    return -(energy_lithiated - energy_host - num_li * energy_li_metal) / num_li


def volume_change_pct(volume_host: float, volume_lithiated: float) -> float:
    """Percent volume change on lithiation, relative to the delithiated host."""
    return (volume_lithiated - volume_host) / volume_host * 100.0


def min_li_clearance(structure: Structure) -> Optional[float]:
    """Shortest Li-to-framework distance, i.e. the tightest gate in the channel.

    ponytail: a true bottleneck needs a percolation path search across the whole
    cell. This is the local floor, which is what actually blocks insertion, and it
    is one line. Swap in a path analysis if the ranking turns out to disagree with
    measured diffusivity.
    """
    li_sites = [s for s in structure if s.specie.symbol == "Li"]
    framework = [s for s in structure if s.specie.symbol != "Li"]
    if not li_sites or not framework:
        return None
    return min(
        structure.lattice.get_distance_and_image(li.frac_coords, other.frac_coords)[0]
        for li in li_sites
        for other in framework
    )


def framework_only(structure: Structure) -> Structure:
    """The structure with every Li stripped out, i.e. the host framework alone."""
    host = structure.copy()
    host.remove_species(["Li"])
    return host


def li_removal_order(structure: Structure) -> list[int]:
    """Site indices of Li, ordered so each removal leaves the rest maximally spread.

    Real delithiation empties the sites that are most crowded first, because two
    close Li repel. So at each rung we drop the Li whose nearest Li neighbour is
    closest.

    ponytail: a greedy spread heuristic, not the lowest-energy Li ordering. The
    honest version enumerates orderings (or runs an Ewald sum) per rung, which is
    exponential in Li count. Upgrade path if a candidate's voltage curve looks
    wrong: score each rung's candidate removals with CHGNet and keep the best.
    """
    li_indices = [i for i, site in enumerate(structure) if site.specie.symbol == "Li"]
    if len(li_indices) < 2:
        return li_indices

    remaining = list(li_indices)
    order = []
    while len(remaining) > 1:
        # Nearest-other-Li distance for each remaining Li; smallest is most crowded.
        crowding = [
            min(structure.get_distance(i, j) for j in remaining if j != i) for i in remaining
        ]
        victim = remaining[min(range(len(remaining)), key=lambda k: (crowding[k], remaining[k]))]
        order.append(victim)
        remaining.remove(victim)
    order.extend(remaining)
    return order


def delithiation_ladder(lithiated: Structure) -> list[tuple[int, Structure]]:
    """Every partial state from fully lithiated down to the bare host.

    Returns [(num_li, structure), ...] descending, so a 3-Li cathode gives
    Li3 -> Li2 -> Li1 -> Li0. A cathode has to survive all of them, not just the
    two endpoints - frameworks usually collapse at an intermediate composition.
    """
    order = li_removal_order(lithiated)
    rungs = [(len(order), lithiated)]
    current = lithiated
    for site_index in order:
        # Indices shift as sites are deleted, so re-find this Li by position.
        current = current.copy()
        target = min(
            range(len(current)),
            key=lambda i: (
                current[i].specie.symbol != "Li",
                current.lattice.get_distance_and_image(
                    current[i].frac_coords, lithiated[site_index].frac_coords
                )[0],
            ),
        )
        current.remove_sites([target])
        rungs.append((len([s for s in current if s.specie.symbol == "Li"]), current))
    return rungs


def find_pairs(pairs_dir: Path) -> list[tuple[int, Path, Path]]:
    """Match `NN_host_*.cif` with `NN_lithiated_*.cif` by their index prefix."""
    hosts = {}
    lithiated = {}
    for cif in sorted(pairs_dir.glob("*.cif")):
        match = re.match(r"(\d+)_(host|lithiated)_", cif.name)
        if not match:
            continue
        (hosts if match.group(2) == "host" else lithiated)[int(match.group(1))] = cif
    return [(i, hosts[i], lithiated[i]) for i in sorted(hosts) if i in lithiated]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_dir", type=Path, default=Path("results/pairs"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/cycling"))
    parser.add_argument("--max_volume_change", type=float, default=MAX_VOLUME_CHANGE_PCT)
    parser.add_argument(
        "--max_step_volume_change",
        type=float,
        default=MAX_STEP_VOLUME_CHANGE_PCT,
        help="largest tolerated volume jump between two adjacent partial states",
    )
    parser.add_argument("--min_voltage", type=float, default=MIN_VOLTAGE)
    parser.add_argument("--max_voltage", type=float, default=MAX_VOLTAGE)
    parser.add_argument("--fmax", type=float, default=0.05, help="force convergence, eV/A")
    parser.add_argument("--steps", type=int, default=500, help="max relaxation steps")
    args = parser.parse_args()

    pairs = find_pairs(args.pairs_dir)
    if not pairs:
        raise SystemExit(f"No NN_host_*.cif / NN_lithiated_*.cif pairs found in {args.pairs_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so --help works without loading CHGNet.
    from chgnet.model.dynamics import StructOptimizer
    from chgnet.model.model import CHGNet

    print(f"Found {len(pairs)} pairs. Loading CHGNet...")
    chgnet = CHGNet.load()
    relaxer = StructOptimizer(model=chgnet)

    def relax(structure: Structure) -> tuple[Structure, float]:
        """Relax and return (structure, total energy in eV)."""
        out = relaxer.relax(structure, fmax=args.fmax, steps=args.steps, verbose=False)
        relaxed = out["final_structure"]
        energy_per_atom = float(chgnet.predict_structure(relaxed)["e"])
        return relaxed, energy_per_atom * len(relaxed)

    li_metal = Structure(Lattice.cubic(LI_METAL_LATTICE), ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    li_relaxed, li_energy_total = relax(li_metal)
    energy_li_metal = li_energy_total / len(li_relaxed)
    print(f"Li metal reference: {energy_li_metal:.4f} eV/atom\n")

    matcher = StructureMatcher()
    rung_rows = []
    summary_rows = []
    for idx, host_cif, lithiated_cif in pairs:
        lithiated_initial = Structure.from_file(lithiated_cif)
        rungs = delithiation_ladder(lithiated_initial)
        if len(rungs) < 2:
            print(f"[{idx:02d}] skipped: no Li to remove")
            continue

        print(f"[{idx:02d}] {lithiated_initial.composition.reduced_formula}: "
              f"{len(rungs)} rungs, {rungs[0][0]} Li -> 0 Li")

        # Relax every partial state. The reference framework for the topotactic
        # test is the fully lithiated one, relaxed: that is the structure we claim
        # to have designed, and every rung has to still be it.
        relaxed = []
        for num_li, rung in rungs:
            structure, energy = relax(rung)
            relaxed.append((num_li, structure, energy))
        reference_framework = framework_only(relaxed[0][1])
        volume_full = relaxed[0][1].volume

        worst_step_dv = 0.0
        broke_at = None
        voltages = []
        for position, (num_li, structure, energy) in enumerate(relaxed):
            fraction_removed = (relaxed[0][0] - num_li) / relaxed[0][0] * 100.0
            cumulative_dv = volume_change_pct(volume_full, structure.volume)
            step_dv = (
                volume_change_pct(relaxed[position - 1][1].volume, structure.volume)
                if position > 0
                else 0.0
            )
            topotactic = bool(matcher.fit(reference_framework, framework_only(structure)))

            # Voltage of the step that produced this rung, i.e. the plateau between
            # this composition and the previous, more lithiated one.
            step_voltage = float("nan")
            if position > 0:
                previous_li, _, previous_energy = relaxed[position - 1]
                delta_n = previous_li - num_li
                if delta_n > 0:
                    step_voltage = average_voltage(
                        previous_energy, energy, delta_n, energy_li_metal
                    )
                    voltages.append(step_voltage)

            worst_step_dv = max(worst_step_dv, abs(step_dv))
            if not topotactic and broke_at is None:
                broke_at = num_li

            structure.to(
                filename=str(args.output_dir / f"{idx:02d}_li{num_li}.cif")
            )
            rung_rows.append(
                {
                    "pair": idx,
                    "num_li": num_li,
                    "pct_li_removed": round(fraction_removed, 1),
                    "formula": structure.composition.reduced_formula,
                    "step_voltage_V": round(step_voltage, 3),
                    "step_volume_change_pct": round(step_dv, 2),
                    "cumulative_volume_change_pct": round(cumulative_dv, 2),
                    "topotactic": topotactic,
                    "min_li_clearance_A": (
                        round(c, 3) if (c := min_li_clearance(structure)) else None
                    ),
                    "energy_eV": round(energy, 3),
                }
            )
            print(
                f"      Li{num_li}  {fraction_removed:5.1f}% removed  "
                f"{structure.composition.reduced_formula:<16} "
                f"V {step_voltage:5.2f}  step dV {step_dv:+6.1f}%  "
                f"total dV {cumulative_dv:+6.1f}%  "
                f"{'ok' if topotactic else 'RECONSTRUCTED'}"
            )

        total_dv = volume_change_pct(volume_full, relaxed[-1][1].volume)
        reasons = []
        if broke_at is not None:
            reasons.append(f"framework reconstructed at Li{broke_at}")
        if abs(total_dv) > args.max_volume_change:
            reasons.append(f"total volume change {total_dv:+.1f}%")
        if worst_step_dv > args.max_step_volume_change:
            reasons.append(f"worst single step {worst_step_dv:.1f}%")
        if voltages and not all(args.min_voltage <= v <= args.max_voltage for v in voltages):
            reasons.append(
                f"voltage range {min(voltages):.2f}-{max(voltages):.2f} V outside window"
            )

        summary_rows.append(
            {
                "pair": idx,
                "host": Structure.from_file(host_cif).composition.reduced_formula,
                "lithiated": lithiated_initial.composition.reduced_formula,
                "num_li": relaxed[0][0],
                "mean_voltage_V": round(sum(voltages) / len(voltages), 3) if voltages else None,
                "min_step_voltage_V": round(min(voltages), 3) if voltages else None,
                "max_step_voltage_V": round(max(voltages), 3) if voltages else None,
                "total_volume_change_pct": round(total_dv, 2),
                "worst_step_volume_change_pct": round(worst_step_dv, 2),
                "reconstructed_at_li": broke_at,
                "verdict": "PASS" if not reasons else "FAIL",
                "reasons": "; ".join(reasons),
            }
        )
        print(f"      -> {'PASS' if not reasons else 'FAIL: ' + '; '.join(reasons)}\n")

    ladder_path = args.output_dir / "ladder_results.csv"
    summary_path = args.output_dir / "cycling_results.csv"
    pd.DataFrame(rung_rows).to_csv(ladder_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    passed = sum(1 for r in summary_rows if r["verdict"] == "PASS")
    print(f"{passed}/{len(summary_rows)} candidates survived full delithiation.")
    print(f"Per-rung table: {ladder_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
