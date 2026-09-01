import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry


class DirectMPRester:
    """
    Direct REST client for Materials Project API to fetch complete chemical systems
    including all elemental endpoints, binaries, and ternaries in a single query.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.materialsproject.org"
        self.headers = {
            "X-API-KEY": self.api_key,
            "accept": "application/json"
        }

    def get_phase_diagram_entries(self, elements: List[str]) -> List[PDEntry]:
        """
        Queries all entries containing ANY combination of the target elements
        to ensure terminal endpoints and sub-systems are fully present.
        """
        # Comma-separated element query returns the full closed subsystem
        elem_str = ",".join(sorted(elements))
        url = f"{self.base_url}/materials/thermo/"
        params = {
            "elements": elem_str,
            "_fields": "formula_pretty,uncorrected_energy,energy_per_atom,composition,is_stable",
            "_limit": 1000
        }
        
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"MP API error ({response.status_code}): {response.text}")
        
        data = response.json().get("data", [])
        elem_set = set(elements)
        entries: List[PDEntry] = []

        for item in data:
            comp_dict = item.get("composition", {})
            # Only keep entries that are pure subsets of our target elements
            if not set(comp_dict.keys()).issubset(elem_set):
                continue

            energy_per_atom = item.get("energy_per_atom", None)
            total_atoms = sum(comp_dict.values())

            if energy_per_atom is not None and total_atoms > 0:
                total_energy = energy_per_atom * total_atoms
                entry = PDEntry(
                    composition=Composition(comp_dict),
                    energy=total_energy,
                    name=item.get("formula_pretty", "Phase")
                )
                entries.append(entry)

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

        total_energy_ev = energy_per_atom * len(struct)
        candidate_entry = PDEntry(
            composition=composition,
            energy=total_energy_ev,
            name=composition.reduced_formula
        )

        try:
            e_above_hull_ev = pd_ref.get_e_above_hull(candidate_entry)
            e_above_hull_mev = e_above_hull_ev * 1000.0

            try:
                decomp_phases, _ = pd_ref.get_decomposition(candidate_entry)
                decomp_str = " + ".join([f"{frac:.2f} {entry.name}" for entry, frac in decomp_phases.items()])
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
                "decomposition_pathway": decomp_str
            }
        except Exception as e:
            print(f"  [WARN] Hull evaluation error for {composition.reduced_formula}: {e}")
            return None


def process_screening_results(
    csv_path: Path,
    cif_dir: Path,
    output_csv: Path,
    api_key: Optional[str] = None
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
                    "decomposition_phases": metrics["decomposition_pathway"]
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
    parser.add_argument(
        "--csv_path", type=str, default="screened_results/screening_results.csv"
    )
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
        Path(args.csv_path),
        Path(args.cif_dir),
        Path(args.output_csv),
        args.api_key
    )