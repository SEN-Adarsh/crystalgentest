import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer


def relax_and_evaluate(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_files = sorted(list(input_dir.glob("*.cif")))
    print(f"Loading {len(cif_files)} lithiated structures from: {input_dir}")

    # Load pretrained CHGNet Universal Potential
    print("Initializing CHGNet potential and optimizer...")
    chgnet = CHGNet.load()
    relaxer = StructOptimizer(model=chgnet)

    results = []

    for cif in cif_files:
        try:
            struct = Structure.from_file(str(cif))
            initial_vol = struct.volume
            formula = struct.composition.reduced_formula

            # Run structure relaxation (coordinates + lattice stress)
            relaxation_result = relaxer.relax(
                struct,
                fmax=0.05,        # Force convergence threshold (eV/Å)
                steps=500,        # Max optimization steps
                verbose=False
            )

            relaxed_struct = relaxation_result["final_structure"]
            final_vol = relaxed_struct.volume
            vol_change = ((final_vol - initial_vol) / initial_vol) * 100.0

            # Calculate energy per atom
            prediction = chgnet.predict_structure(relaxed_struct)
            energy_per_atom = float(prediction["e"])

            # Save relaxed structure CIF
            out_cif = output_dir / f"relaxed_{cif.name}"
            relaxed_struct.to(filename=str(out_cif))

            # Theoretical capacity estimate (Faraday: C = (n * F) / (3.6 * M))
            n_li = len([s for s in relaxed_struct if s.specie.symbol == "Li"])
            molar_mass = relaxed_struct.composition.weight
            faraday = 96485.33  # C/mol
            c_theory = (n_li * faraday) / (3.6 * molar_mass) if molar_mass > 0 else 0.0

            results.append({
                "file": cif.name,
                "formula": formula,
                "total_atoms": len(relaxed_struct),
                "energy_per_atom_eV": round(energy_per_atom, 4),
                "vol_change_pct": round(vol_change, 2),
                "theoretical_capacity_mAh_g": round(c_theory, 1),
                "status": "Converged"
            })

            print(f"[RELAXED] {formula:<18} | Energy: {energy_per_atom:.3f} eV/atom | Cap: {c_theory:.1f} mAh/g | ΔV: {vol_change:+.1f}%")

        except Exception as e:
            print(f"[FAILED] {cif.name}: {e}")
            results.append({
                "file": cif.name,
                "formula": "N/A",
                "total_atoms": 0,
                "energy_per_atom_eV": np.nan,
                "vol_change_pct": np.nan,
                "theoretical_capacity_mAh_g": 0.0,
                "status": f"Error: {e}"
            })

    # Export structured ranking table
    df = pd.DataFrame(results)
    csv_path = output_dir / "screening_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nScreening complete! Summary table saved to:\n{csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHGNet Relaxation & Capacity Screening")
    parser.add_argument(
        "--input_dir", type=str, default="results", help="Path to unrelaxed lithiated CIFs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="screened_results",
        help="Path to save relaxed CIFs and CSV",
    )
    args = parser.parse_args()

    relax_and_evaluate(Path(args.input_dir), Path(args.output_dir))