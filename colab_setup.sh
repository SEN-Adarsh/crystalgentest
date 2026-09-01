#!/usr/bin/env bash
# Colab setup. Run once per session:  !bash colab_setup.sh
#
# Colab's own runtime is Python 3.13 with numpy 2.x, and `pip install -e .`
# cannot satisfy this project there: torch 2.2.1+cu118 has no cp313 wheel,
# torch_scatter/sparse/cluster have no cp313 wheels for it either, and pip
# ignores the [tool.uv.sources] block where upstream pinned those wheels by URL.
# So build a Python 3.10 venv with uv instead, which does honour that block.
#
# Everything afterwards runs through `uv run`, e.g.
#   !uv run python -m mattergen.cycling_screen --pairs_dir results/pairs
#
# ponytail: a 3.10 venv, not a de-pinned 3.13 install. Relaxing the pins means
# porting gemnet off torch_scatter onto native torch.scatter_reduce - real work,
# and it changes numerics. Revisit if upstream drops the pins.
set -euo pipefail

pip install -q uv
uv venv --python 3.10
uv sync

# CHGNet drives the cycling screen and is not a project dependency. It declares
# torch>=2.4.1 but runs fine on 2.2.1, so install it without letting it pull a
# different torch in and break the CUDA build.
uv pip install --no-deps chgnet
uv pip install pymatgen ase typing-extensions

uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run python -c "import torch_scatter, torch_sparse, torch_cluster; print('pyg extensions ok')"
uv run python test_pipeline.py
