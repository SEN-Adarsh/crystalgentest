#!/usr/bin/env bash
# Colab setup. Run once per session:  !bash colab_setup.sh
#
# Installs against the torch Colab already ships, on Colab's own Python. The
# only thing pip cannot work out by itself is torch_scatter / torch_sparse /
# torch_cluster: PyPI carries source-only, so pip would compile them for ~20
# minutes. data.pyg.org has prebuilt wheels, but they are keyed to one exact
# torch version, so the find-links URL has to be derived at runtime.
set -euo pipefail

TORCH=$(python -c "import torch; print(torch.__version__)")
echo "building against torch $TORCH"

pip install -q -e . -f "https://data.pyg.org/whl/torch-${TORCH}.html"

# CHGNet drives the cycling screen and is not a project dependency. --no-deps
# keeps it from pulling a different torch in and breaking the CUDA build.
pip install -q --no-deps chgnet

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torch_scatter, torch_sparse, torch_cluster; print('pyg extensions ok')"
python test_pipeline.py
