#!/usr/bin/env bash
# Colab setup. Run this INSTEAD of `pip install -e .`:  !bash colab_setup.sh
#
# The one thing pip cannot work out alone is torch_scatter / torch_sparse /
# torch_cluster: PyPI carries source only, so pip compiles each for 15-20
# minutes. data.pyg.org ships them prebuilt, but keyed to one exact torch build,
# so the find-links URL has to be derived at runtime.
#
# Colab's default torch moves every few weeks and runs ahead of what
# data.pyg.org publishes. When it does, downgrade torch to the newest build PyG
# has wheels for rather than compiling or hardcoding a guess that rots again.
set -euo pipefail

has_wheels() { curl -sfI "https://data.pyg.org/whl/torch-$1.html" > /dev/null; }

TORCH=$(python -c "import torch; print(torch.__version__)")
echo "found torch $TORCH"

if ! has_wheels "$TORCH"; then
  echo "data.pyg.org has nothing for torch $TORCH; searching for a version it does have"
  PINNED=""
  for v in 2.11.0 2.10.0 2.9.1 2.9.0 2.8.0 2.7.1 2.7.0 2.6.0 2.5.1 2.5.0 2.4.1; do
    for cu in cu128 cu126 cu124 cu121; do
      if has_wheels "${v}+${cu}"; then PINNED="$v"; CUDA="$cu"; break 2; fi
    done
  done
  if [ -z "$PINNED" ]; then
    echo "no candidate torch has PyG wheels; check https://data.pyg.org/whl/ by hand" >&2
    exit 1
  fi
  echo "pinning torch==$PINNED ($CUDA) -- roughly a 2 GB download"
  pip install -q "torch==$PINNED" --index-url "https://download.pytorch.org/whl/$CUDA"
  TORCH=$(python -c "import torch; print(torch.__version__)")
  echo "now on torch $TORCH"
fi

WHEELS="https://data.pyg.org/whl/torch-${TORCH}.html"

# Pin torch on the install line too. pyproject leaves it deliberately unbounded,
# so without this pip is free to drag it back to a version with no wheels.
pip install -e . -f "$WHEELS" "torch==${TORCH%%+*}"

# CHGNet drives the cycling screen and is not a project dependency. --no-deps
# keeps it from pulling a different torch in and breaking the CUDA build.
pip install --no-deps chgnet

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torch_scatter, torch_sparse, torch_cluster; print('pyg extensions ok')"
python test_pipeline.py
