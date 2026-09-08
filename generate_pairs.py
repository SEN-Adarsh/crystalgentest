"""Generate host/lithiated CIF pairs for viewing in VESTA.

Stage 1 samples delithiated host frameworks from the diffusion model; stage 2
places Li deterministically into each host's interstitial voids. Both members of
each pair are written as CIFs so the insertion can be inspected side by side.

Hosts with no redox-active metal (or no charge headroom) accept no Li and are
skipped, so we oversample until enough pairs are found.

Usage:
    python generate_pairs.py --num_pairs 5
"""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from pymatgen.core import Structure

from crystalgen.common.utils.eval_utils import CrystalGenCheckpointInfo
from crystalgen.generator import CrystalGenerator
from crystalgen.li_placer import PhysicsInformedLiPlacer

DEFAULT_CHECKPOINT = Path("checkpoints/base_model/checkpoints/mattergen_base")
DEFAULT_OUTPUT_DIR = Path("results/pairs")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _write_provenance(args: argparse.Namespace, output_dir: Path) -> None:
    """Stamp every generation run with what produced it (checkpoint, guidance
    weight, commit). v2/v3 batches are unrecoverable without this."""
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": str(args.checkpoint.resolve()),
        "guidance_weight": args.guidance_weight,
        "batch_size": args.batch_size,
        "num_pairs": args.num_pairs,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    (output_dir / "_provenance.json").write_text(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_rounds", type=int, default=4)
    # Feature 3: steer reverse diffusion towards open, corner-sharing polyhedral
    # networks. 0 reproduces stock sampling; needs no retraining.
    parser.add_argument("--guidance_weight", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_provenance(args, args.output_dir)
    scratch = args.output_dir / "_sampling"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sampling device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    generator = CrystalGenerator(
        checkpoint_info=CrystalGenCheckpointInfo(args.checkpoint.resolve()),
        batch_size=args.batch_size,
        num_batches=1,
        record_trajectories=False,
        sampling_config_overrides=[
            f"sampler_partial.polyhedral_guidance_weight={args.guidance_weight}"
        ],
    )
    placer = PhysicsInformedLiPlacer()

    started = time.perf_counter()
    pairs: list[tuple[Structure, Structure]] = []
    for round_idx in range(args.max_rounds):
        round_started = time.perf_counter()
        # CrystalGen writes its own extxyz dump into this directory and does not
        # create it itself.
        scratch.mkdir(parents=True, exist_ok=True)
        hosts = generator.generate(output_dir=str(scratch), hierarchical_lithiation=False)
        sampling_time = time.perf_counter() - round_started

        for host in hosts:
            lithiated = placer.place_lithium(host)
            if lithiated is None:
                continue
            pairs.append((host, lithiated))
            if len(pairs) == args.num_pairs:
                break

        print(
            f"Round {round_idx + 1}: sampled {len(hosts)} hosts in {sampling_time / 60:.1f} min "
            f"({sampling_time / max(len(hosts), 1):.1f} s/structure), "
            f"{len(pairs)}/{args.num_pairs} pairs so far.",
            flush=True,
        )
        if len(pairs) == args.num_pairs:
            break

    elapsed = time.perf_counter() - started
    shutil.rmtree(scratch, ignore_errors=True)

    if not pairs:
        raise SystemExit("No sampled host accepted Li. Try more rounds or a larger batch.")

    for idx, (host, lithiated) in enumerate(pairs, start=1):
        host_name = host.composition.reduced_formula
        li_name = lithiated.composition.reduced_formula
        host.to(filename=str(args.output_dir / f"{idx:02d}_host_{host_name}.cif"))
        lithiated.to(filename=str(args.output_dir / f"{idx:02d}_lithiated_{li_name}.cif"))
        n_li = sum(1 for s in lithiated if s.specie.symbol == "Li")
        print(f"{idx:02d}  {host_name} -> {li_name}  (+{n_li} Li)")

    print(f"\n{len(pairs)} pairs written to {args.output_dir}")
    print(f"Total wall time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
