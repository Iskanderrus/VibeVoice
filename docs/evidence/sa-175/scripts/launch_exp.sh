#!/usr/bin/env bash
# Wait for any running VibeVoice job (the in-flight v3 take) to exit, then start the scene-2 comparison grid.
while pgrep -f "run_stickman_local.py --foreground" >/dev/null; do sleep 3; done
: > /tmp/claude-1002/vvpre/exp.log
exec ~/.local/share/stickman-vibevoice/venv-rebuild/bin/python /tmp/claude-1002/vvpre/exp_scene2.py > /tmp/claude-1002/vvpre/exp.out 2>&1
