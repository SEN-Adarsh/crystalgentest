"""Fine-tune the pretrained MatterGen diffusion model on delithiated cathode hosts.

Stage 1 of the hierarchical pipeline: the model learns the distribution of
delithiated host frameworks. The physics-informed auxiliary losses (redox window
and octahedral geometry) are applied automatically during training - see
mattergen.diffusion.losses.SummedFieldLoss.

Usage:
    python finetune.py --max_epochs 20
"""

import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo, load_model_diffusion

DEFAULT_MANIFEST = Path("data/delithiated_manifest.json")
DEFAULT_HOSTS_DIR = Path("data/delithiated_hosts")
DEFAULT_BASE_CHECKPOINT = Path("checkpoints/base_model/checkpoints/mattergen_base")
DEFAULT_OUTPUT_DIR = Path("checkpoints/hierarchical_finetuned")


class DelithiatedHostDataset(Dataset):
    """Delithiated host frameworks, read from CIFs listed in a manifest."""

    def __init__(self, manifest_path: Path, hosts_dir: Path):
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.hosts_dir = hosts_dir

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, record: dict) -> str:
        cif_path = record["cif_path"]
        if not os.path.exists(cif_path):
            cif_path = str(self.hosts_dir / os.path.basename(cif_path))
        return cif_path

    def __getitem__(self, idx: int) -> ChemGraph:
        struct = Structure.from_file(self._resolve(self.records[idx]))

        return ChemGraph(
            pos=torch.tensor(struct.frac_coords, dtype=torch.float32),
            atomic_numbers=torch.tensor(struct.atomic_numbers, dtype=torch.long),
            cell=torch.tensor(struct.lattice.matrix, dtype=torch.float32).unsqueeze(0),
            num_atoms=torch.tensor([len(struct)], dtype=torch.long),
        )


def build_dataloaders(
    manifest_path: Path, hosts_dir: Path, batch_size: int, num_workers: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    dataset = DelithiatedHostDataset(manifest_path, hosts_dir)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size

    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    print(f"Dataset ready: {train_size} train structures, {val_size} val structures.")

    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--hosts_dir", type=Path, default=DEFAULT_HOSTS_DIR)
    parser.add_argument("--base_checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # The physics penalties are in units of (valence)^2 and (cos)^4, which are
    # numerically much larger than the score-matching terms; these weights are the
    # calibration knob. Set either to 0 to disable that guidance term.
    parser.add_argument("--redox_weight", type=float, default=None)
    parser.add_argument("--poly_weight", type=float, default=None)
    parser.add_argument("--connectivity_weight", type=float, default=None)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    print("Preparing delithiated dataset loader...")
    train_loader, val_loader = build_dataloaders(
        args.manifest, args.hosts_dir, args.batch_size, args.num_workers, args.seed
    )

    print("\n--- Loading Pretrained Diffusion Model ---")
    # MatterGenCheckpointInfo needs the directory holding config.yaml and the
    # checkpoints/ subfolder, as an absolute path.
    checkpoint_info = MatterGenCheckpointInfo(args.base_checkpoint.resolve())
    model = load_model_diffusion(checkpoint_info)

    loss_fn = model.diffusion_module.loss_fn
    if args.redox_weight is not None:
        loss_fn.redox_weight = args.redox_weight
    if args.poly_weight is not None:
        loss_fn.poly_weight = args.poly_weight
    if args.connectivity_weight is not None:
        loss_fn.connectivity_weight = args.connectivity_weight
    print(
        f"Physics guidance: redox={loss_fn.redox_weight}  poly={loss_fn.poly_weight}  "
        f"connectivity={loss_fn.connectivity_weight}"
    )

    # `loss_val` is logged by DiffusionLightningModule for every validation step;
    # the physics terms appear alongside it as `loss_redox_val` / `loss_poly_val`.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(args.output_dir),
        filename="scaffold-diff-{epoch:02d}-{loss_val:.3f}",
        monitor="loss_val",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="16-mixed" if use_gpu else 32,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        default_root_dir=str(args.output_dir),
    )

    print("\n--- Starting Fine-Tuning ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
