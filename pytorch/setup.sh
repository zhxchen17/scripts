#!/usr/bin/env bash
# Clone PyTorch and install dependencies for development.

set -euo pipefail

gh repo clone pytorch/pytorch "$HOME/pytorch"
cd "$HOME/pytorch"

uv pip install --group dev
python3 -m ensurepip --upgrade
make triton

echo "Run: MAX_JOBS=240 USE_CUDA=1 uv pip install --no-build-isolation -v -e ."
