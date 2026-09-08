"""Paper-1 experiment driver: every ablation arm from one command.

Each arm = one directory under the output root, containing the generated
CIFs, per-structure motif records (motifs.json), the per-step guidance
traces (steer_trace_*.json), and a provenance stamp. Arms are resumable:
an arm whose summary.json already exists is skipped, so a dead Colab
session can be re-run with the same command.

Suites:
    smoke   one small w=1 arm (pipeline verification, ~10 min)
    run0    dose-response: w in {0, 0.5, 1, 2, 5}              (Run 0 + dose)
    run1    per-term: geometry-only, connectivity-only          (Run 1 sampling side)
    run2    base-checkpoint arms: base+w0, base+w1              (Run 2 missing cells)
    run3    annealing off @ w=1, clamped @ w=1                  (Run 3 + clamp)
    parity  w=0 twice at the same seed; CIFs must be identical  (w=0 no-op check)

Usage:
    python run_experiments.py --suite run0 --n 200 --batch_size 64 \
        --checkpoint checkpoints/scaffold --output_dir results/paper1
    python run_experiments.py --suite run0,run1 --arms w1   # subset / resume
    python run_experiments.py --aggregate results/paper1    # summary CSV
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from pymatgen.core import Structure

from crystalgen.common.utils.eval_utils import CrystalGenCheckpointInfo
from crystalgen.diffusion.motifs import motif_records, summarize_records
from crystalgen.generator import CrystalGenerator
from crystalgen.li_placer import PhysicsInformedLiPlacer
from crystalgen.validity import (
    is_smact_valid,
    is_structure_valid,
    load_reference_structures,
    novelty_report,
)


@dataclass
class Arm:
    name: str
    checkpoint: str = "checkpoints/scaffold"
    guidance_weight: float = 1.0
    geometry_weight: float = 1.0
    connectivity_weight: float = 1.0
    annealing: bool = True
    max_guidance_rel: float | None = None
    seed: int = 42
    extra_overrides: dict = field(default_factory=dict)


def base_arms(checkpoint: str) -> dict:
    return {
        "w0": Arm(name="w0", checkpoint=checkpoint, guidance_weight=0.0),
        "w0.5": Arm(name="w0.5", checkpoint=checkpoint, guidance_weight=0.5),
        "w1": Arm(name="w1", checkpoint=checkpoint, guidance_weight=1.0),
        "w2": Arm(name="w2", checkpoint=checkpoint, guidance_weight=2.0),
        "w5": Arm(name="w5", checkpoint=checkpoint, guidance_weight=5.0),
    }


SUITES = {
    "smoke": lambda ckpt: {"w1": Arm(name="w1", checkpoint=ckpt)},
    "run0": lambda ckpt: base_arms(ckpt),
    "run1": lambda ckpt: {
        "geom_only": Arm(
            name="geom_only", checkpoint=ckpt, guidance_weight=1.0, connectivity_weight=0.0
        ),
        "conn_only": Arm(
            name="conn_only", checkpoint=ckpt, guidance_weight=1.0, geometry_weight=0.0
        ),
    },
    "run2": lambda ckpt: {
        "base_w0": Arm(name="base_w0", checkpoint="checkpoints/base", guidance_weight=0.0),
        "base_w1": Arm(name="base_w1", checkpoint="checkpoints/base", guidance_weight=1.0),
    },
    "run3": lambda ckpt: {
        "w1_noanneal": Arm(name="w1_noanneal", checkpoint=ckpt, annealing=False),
        "w1_clamped": Arm(name="w1_clamped", checkpoint=ckpt, max_guidance_rel=0.1),
    },
    "parity": lambda ckpt: {
        "parity_a": Arm(name="parity_a", checkpoint=ckpt, guidance_weight=0.0, seed=7),
        "parity_b": Arm(name="parity_b", checkpoint=ckpt, guidance_weight=0.0, seed=7),
    },
}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _overrides(arm: Arm, arm_dir: Path, record_trace: bool) -> list[str]:
    o = [
        f"sampler_partial.polyhedral_guidance_weight={arm.guidance_weight}",
        f"sampler_partial.polyhedral_geometry_weight={arm.geometry_weight}",
        f"sampler_partial.polyhedral_connectivity_weight={arm.connectivity_weight}",
        f"sampler_partial.polyhedral_annealing={str(arm.annealing).lower()}",
    ]
    if arm.max_guidance_rel is not None:
        o.append(f"sampler_partial.max_guidance_rel={arm.max_guidance_rel}")
    if record_trace:
        o.append(f"sampler_partial.steer_trace_path={arm_dir / 'steer_trace.json'}")
    o.extend(arm.extra_overrides)
    return o


def run_arm(
    arm: Arm,
    generator: CrystalGenerator,
    n: int,
    batch_size: int,
    output_root: Path,
    ref_by_formula: dict | None,
) -> dict:
    arm_dir = output_root / arm.name
    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        print(f"[{arm.name}] summary exists, skipping (resume)")
        return json.loads(summary_path.read_text())

    arm_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # Same seed for every arm -> identical atom-count conditioning and noise
    # draws; arms differ only through the guidance term (paired design).
    torch.manual_seed(arm.seed)

    n_batches = (n + batch_size - 1) // batch_size
    generator.sampling_config_overrides = _overrides(
        arm, arm_dir, record_trace=arm.guidance_weight > 0.0
    )
    hosts = generator.generate(
        batch_size=batch_size, num_batches=n_batches, hierarchical_lithiation=False
    )
    hosts = hosts[:n]

    names = []
    for i, h in enumerate(hosts):
        name = f"{i:04d}_{h.composition.reduced_formula}"
        h.to(filename=str(arm_dir / f"{name}.cif"))
        names.append(name)

    records = motif_records(hosts, names=names)
    (arm_dir / "motifs.json").write_text(json.dumps(records, indent=1))
    summary = summarize_records(records)

    # Run-4 style screens on the same structures (cheap, no CHGNet here).
    summary["structure_valid_frac"] = sum(is_structure_valid(s) for s in hosts) / len(hosts)
    summary["smact_valid_frac"] = sum(is_smact_valid(s) for s in hosts) / len(hosts)
    if ref_by_formula is not None:
        nov = novelty_report(hosts, ref_by_formula)
        summary["novel_frac"] = sum(r["novel"] for r in nov) / len(nov)
        (arm_dir / "novelty.json").write_text(json.dumps(nov, indent=1))

    # Li-acceptance rate (pipeline-level outcome, not just geometry).
    placer = PhysicsInformedLiPlacer()
    summary["li_accept_frac"] = sum(placer.place_lithium(s) is not None for s in hosts) / len(hosts)

    summary["arm"] = asdict(arm)
    summary["n_generated"] = len(hosts)
    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    summary["commit"] = _git_commit()
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"[{arm.name}] done in {summary['wall_time_s']}s: {summary}")
    return summary


def check_parity(output_root: Path) -> bool:
    a_dir, b_dir = output_root / "parity_a", output_root / "parity_b"
    a = sorted(p.name for p in a_dir.glob("*.cif"))
    b = sorted(p.name for p in b_dir.glob("*.cif"))
    if a != b:
        print(f"[parity] FAIL: file lists differ")
        return False
    for name in a:
        if (a_dir / name).read_text() != (b_dir / name).read_text():
            print(f"[parity] FAIL: {name} differs between runs")
            return False
    print(f"[parity] PASS: {len(a)} CIFs bit-for-bit identical at w=0")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=str, default="smoke", help="comma-separated suites")
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm subset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scaffold")
    parser.add_argument("--n", type=int, default=200, help="structures per arm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper1"))
    parser.add_argument(
        "--reference_dir",
        type=Path,
        default=Path("data/delithiated_hosts"),
        help="reference structures for the novelty screen",
    )
    parser.add_argument("--aggregate", action="store_true", help="only write the summary CSV")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        aggregate(args.output_dir)
        return

    suite_names = args.suite.split(",")
    arms: dict[str, Arm] = {}
    for s in suite_names:
        if s not in SUITES:
            raise SystemExit(f"unknown suite {s!r}; choose from {sorted(SUITES)}")
        arms.update(SUITES[s](args.checkpoint))
    if args.arms:
        keep = set(args.arms.split(","))
        missing = keep - set(arms)
        if missing:
            raise SystemExit(f"arms {sorted(missing)} not in suites {suite_names}")
        arms = {k: v for k, v in arms.items() if k in keep}

    # One model load per checkpoint, reused across arms. Same for the
    # novelty reference set (4544 CIFs parsed once, not once per arm).
    generators: dict[str, CrystalGenerator] = {}
    ref_by_formula = None
    if args.reference_dir.exists():
        print(f"loading novelty reference set from {args.reference_dir} ...")
        ref_by_formula = load_reference_structures(args.reference_dir)
        print(
            f"  {sum(len(v) for v in ref_by_formula.values())} structures, "
            f"{len(ref_by_formula)} formulas"
        )

    def get_generator(ckpt: str) -> CrystalGenerator:
        if ckpt not in generators:
            generators[ckpt] = CrystalGenerator(
                checkpoint_info=CrystalGenCheckpointInfo(Path(ckpt).resolve()),
                batch_size=args.batch_size,
                num_batches=1,
                record_trajectories=False,
            )
        return generators[ckpt]

    for arm in arms.values():
        run_arm(
            arm,
            get_generator(arm.checkpoint),
            args.n,
            args.batch_size,
            args.output_dir,
            ref_by_formula,
        )

    if "parity" in suite_names:
        ok = check_parity(args.output_dir)
        (args.output_dir / "parity_result.txt").write_text("PASS" if ok else "FAIL")

    aggregate(args.output_dir)


def aggregate(output_root: Path) -> None:
    rows = []
    for summary_path in sorted(output_root.glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        row = {"arm": s["arm"]["name"]}
        for k in (
            "n_generated",
            "n_scored",
            "scored_frac",
            "corner_frac",
            "edge_frac",
            "face_frac",
            "neighbours_per_structure",
            "mean_cation_cn",
            "corner_pairs",
            "edge_pairs",
            "face_pairs",
            "structure_valid_frac",
            "smact_valid_frac",
            "novel_frac",
            "li_accept_frac",
            "wall_time_s",
            "commit",
        ):
            row[k] = s.get(k)
        rows.append(row)
    if not rows:
        print("no completed arms found")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    out = output_root / "aggregate.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
