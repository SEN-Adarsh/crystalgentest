"""Paper-1 Phase C: statistical analysis of run_experiments output.

Reads results/<root>/<arm>/motifs.json per arm, computes:
  - arm-level motif fractions with bootstrap CIs (structure-level resampling)
  - paired per-structure deltas vs the w0 arm (same seed -> same conditioning
    index; the paired comparison cancels conditioning variance)
  - dose-response table (corner_frac vs w)
  - the scored-asymmetry audit (scored_frac, neighbours_per_structure)

Usage:
    python analyze_runs.py --root results/paper1
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(root: Path, arm: str):
    p = root / arm / "motifs.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def boot_ci(values, n_boot=2000, seed=0, stat=np.mean):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.array(
        [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    )
    return (
        float(stat(values)),
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def arm_stats(records):
    """Per-structure corner fractions + absolute counts for scored structures."""
    scored = [r for r in records if r["scored"]]
    corner_frac = [r["corner_frac"] for r in scored]
    corner_pairs = [r["corner_pairs"] for r in scored]
    edge_pairs = [r["edge_pairs"] for r in scored]
    face_pairs = [r["face_pairs"] for r in scored]
    cn = [r["mean_cation_cn"] for r in scored]
    return {
        "n": len(records),
        "n_scored": len(scored),
        "scored_frac": len(scored) / len(records) if records else float("nan"),
        "corner_frac": corner_frac,
        "corner_pairs": corner_pairs,
        "edge_pairs": edge_pairs,
        "face_pairs": face_pairs,
        "mean_cation_cn": cn,
        "total_pairs": [r["total_neighbours"] for r in scored],
    }


def paired_delta(records_a, records_b):
    """Per-index (same conditioning slot) corner-pair delta: b - a."""
    n = min(len(records_a), len(records_b))
    out = []
    for i in range(n):
        ra, rb = records_a[i], records_b[i]
        if ra["scored"] and rb["scored"]:
            out.append(
                (rb["corner_pairs"] - ra["corner_pairs"], rb["edge_pairs"] - ra["edge_pairs"])
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/paper1"))
    parser.add_argument("--baseline", type=str, default="w0")
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    arms = sorted(d.name for d in args.root.iterdir() if (d / "motifs.json").exists())
    if not arms:
        raise SystemExit(f"no completed arms under {args.root}")
    print(f"arms: {arms}\n")

    all_stats = {a: arm_stats(load_records(args.root, a)) for a in arms}

    print("=== arm-level, structure-resampled bootstrap 95% CI ===")
    hdr = (
        f"{'arm':<12}{'n':>4}{'scrd':>6}{'corner% (mean [CI])':>28}"
        f"{'edge/struct':>13}{'CN':>7}{'scored%':>9}"
    )
    print(hdr)
    for a, s in all_stats.items():
        mean, lo, hi = boot_ci(s["corner_frac"], n_boot=args.n_boot)
        edge_per = np.mean(s["edge_pairs"]) if s["edge_pairs"] else float("nan")
        cn = np.mean(s["mean_cation_cn"]) if s["mean_cation_cn"] else float("nan")
        print(
            f"{a:<12}{s['n']:>4}{s['n_scored']:>6}"
            f"{100 * mean:>10.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
            f"{edge_per:>13.1f}{cn:>7.2f}{100 * s['scored_frac']:>9.1f}"
        )

    if args.baseline in all_stats:
        base = load_records(args.root, args.baseline)
        print(f"\n=== paired per-structure deltas vs {args.baseline} (same conditioning slot) ===")
        print(
            f"{'arm':<12}{'n pairs':>8}{'d(corner/struct)':>18}{'d(edge/struct)':>16}{'CI d(corner)':>26}"
        )
        for a in arms:
            if a == args.baseline:
                continue
            deltas = paired_delta(base, load_records(args.root, a))
            if not deltas:
                print(f"{a:<12}{'0':>8}  (no jointly scored structures)")
                continue
            dcorner = [d[0] for d in deltas]
            dedge = [d[1] for d in deltas]
            mean, lo, hi = boot_ci(dcorner, n_boot=args.n_boot)
            print(
                f"{a:<12}{len(deltas):>8}{np.mean(dcorner):>18.2f}{np.mean(dedge):>16.2f}"
                f"  [{lo:.2f}, {hi:.2f}]"
            )

    # Dose-response if the arms are a w grid
    try:
        ws = sorted(
            (a for a in arms if a.startswith("w") and a[1:].replace(".", "", 1).isdigit()),
            key=lambda a: float(a[1:]),
        )
    except ValueError:
        ws = []
    if len(ws) >= 3:
        print("\n=== dose-response (aggregate pool) ===")
        for a in ws:
            s = all_stats[a]
            tot_c = sum(s["corner_pairs"])
            tot_e = sum(s["edge_pairs"])
            tot_f = sum(s["face_pairs"])
            tot = tot_c + tot_e + tot_f
            print(
                f"w={a[1:]:<6} corner={100 * tot_c / max(tot, 1):.1f}%  "
                f"edge={100 * tot_e / max(tot, 1):.1f}%  scored={100 * s['scored_frac']:.1f}%  "
                f"pairs/struct={np.mean(s['total_pairs']) if s['total_pairs'] else float('nan'):.1f}"
            )


if __name__ == "__main__":
    main()
