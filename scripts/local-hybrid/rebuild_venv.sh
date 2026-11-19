#!/bin/bash
# Controlled rebuild of the runtime venv. Every parameter is set here, and the
# deviations from requirements-stickman.lock are explicit (see DEVIATIONS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FORK="$(cd "$HERE/../.." && pwd)"
VENV="${1:-$HOME/.local/share/stickman-vibevoice/venv-rebuild}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
PYVER="${PYVER:-3.11}"
# Service-only lock excludes unused WebRTC demo dependencies.
# No manual lock filtering, venv patching or editable source overrides.
grep -v -E '^(av==14\.3\.0|aiortc==1\.13\.0)$' "$FORK/requirements-stickman.lock" > "$VENV.lock.txt"
uv venv --python "$PYVER" "$VENV"
uv pip install --python "$VENV/bin/python" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
uv pip install --python "$VENV/bin/python" -r "$VENV.lock.txt"
uv pip install --python "$VENV/bin/python" pytest==8.4.1
uv pip check --python "$VENV/bin/python"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
echo "venv ready: $VENV/bin/python"
