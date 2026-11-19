#!/usr/bin/env bash
# After the running batch: first scene 2 / seed 18430 (needed for the seed-18430 episode), then the whole-piece comparison.
while pgrep -f "batch_scenes.py" >/dev/null || pgrep -f "run_stickman_local.py --foreground" >/dev/null; do sleep 5; done
bash /tmp/claude-1002/vvpre/one_take_s2_18430.sh
while pgrep -f "run_stickman_local.py --foreground" >/dev/null; do sleep 5; done
: > /tmp/claude-1002/vvpre/whole.log
exec ~/.local/share/stickman-vibevoice/venv-rebuild/bin/python /tmp/claude-1002/vvpre/whole_piece.py > /tmp/claude-1002/vvpre/whole.out 2>&1
