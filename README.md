# CrystalGen

Diffusion-based generation of Li-ion battery cathode materials.

CrystalGen generates delithiated host scaffolds with a fine-tuned crystal
diffusion model, steers them toward open, corner-sharing polyhedral networks,
inserts Li deterministically into charge-balanced interstitial sites, and
screens the results with CHGNet relaxation.

Pipeline: **generate host scaffold -> insert Li -> CHGNet relax -> screen**

## What's in it

- `crystalgen/diffusion/polyhedra.py` — polyhedral geometry + corner-sharing
  connectivity losses (bridging-anion formulation, differentiable)
- `crystalgen/diffusion/redox.py` — bond-valence-sum redox-window loss with
  charge neutrality
- `crystalgen/diffusion/physics.py` — shared first-shell periodic graph for
  the physics losses (3.2 Å / 20 neighbours)
- `crystalgen/li_placer.py` — deterministic physics-informed Li placer
  (Voronoi voids, Coulomb ranking, charge-balanced greedy insertion)
- `crystalgen/diffusion/motifs.py` — corner/edge/face motif statistics
- `generate_pairs.py` — host+lithiated CIF pair generation (provenance-stamped)
- `finetune.py` — fine-tune the base model on delithiated hosts with the
  physics losses enabled
- `relax_and_screen.py`, `screen_ehull.py`, `cycling_screen.py` — CHGNet
  relaxation, theoretical-capacity, and convex-hull screening

## Setup (Colab)

```bash
!git clone https://github.com/SEN-Adarsh/crystalgentest.git /content/crystalgentest
%cd /content/crystalgentest
!bash colab_setup.sh
```

`colab_setup.sh` installs the package plus the PyG extension wheels matching
the runtime's torch, then runs `test_pipeline.py` (physics smoke checks).

Checkpoints: stage `config.yaml` + `checkpoints/last.ckpt` under
`checkpoints/scaffold` (fine-tuned) or `checkpoints/base_model/checkpoints/mattergen_base`
(base). The fine-tuned checkpoint config from Hugging Face references the old
package name; patch it after download:

```python
text = Path("checkpoints/scaffold/config.yaml").read_text().replace("mattergen", "crystalgen")
Path("checkpoints/scaffold/config.yaml").write_text(text)
```

## Usage

```bash
# host + lithiated pairs, polyhedral guidance on
python generate_pairs.py --checkpoint checkpoints/scaffold \
  --num_pairs 10 --batch_size 32 --guidance_weight 1.0 --output_dir results/pairs

# fine-tune the base model on the delithiated-host dataset
python finetune.py --max_epochs 20
```

Every generation run writes `_provenance.json` (commit, checkpoint, guidance
weight, torch build).

## Data

`data/delithiated_manifest.json` + `data/delithiated_hosts/` — 4544
delithiated cathode host frameworks derived from Materials Project structures.

## Credits

Derived from [Microsoft MatterGen](https://github.com/microsoft/mattergen)
(MIT license); see LICENSE. The base GemNet-T denoiser and diffusion
scaffolding are upstream; the cathode pipeline (polyhedral guidance, redox
loss, Li placer, screening) is this project's contribution.
