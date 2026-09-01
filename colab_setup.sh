#!/usr/bin/env bash
# Colab setup. Run this INSTEAD of `pip install -e .`:  !bash colab_setup.sh
#
# The one thing pip cannot work out alone is torch_scatter / torch_sparse /
# torch_cluster: PyPI carries source only, so pip compiles each for 15-20
# minutes. data.pyg.org ships them prebuilt, but keyed to one exact torch build,
# so the find-links URL has to be derived at runtime.
set -euo pipefail

TORCH=$(python -c "import torch; print(torch.__version__)")
WHEELS="https://data.pyg.org/whl/torch-${TORCH}.html"
echo "building against torch $TORCH"

# Fail loudly rather than let pip fall through to a source build.
if ! curl -sfI "$WHEELS" > /dev/null; then
  echo "no prebuilt PyG extensions for torch $TORCH at $WHEELS" >&2
  echo "pick a torch that data.pyg.org publishes for, e.g." >&2
  echo "  pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128" >&2
  exit 1
fi

pip install -q -e . -f "$WHEELS"

# CHGNet drives the cycling screen and is not a project dependency. --no-deps
# keeps it from pulling a different torch in and breaking the CUDA build.
pip install -q --no-deps chgnet

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torch_scatter, torch_sparse, torch_cluster; print('pyg extensions ok')"
python test_pipeline.py
