#!/usr/bin/env bash
# Wait for the scene-2 grid and any running VibeVoice job, then run the pitch-shifted-reference comparison.
while pgrep -f "exp_scene2.py" >/dev/null || pgrep -f "run_stickman_local.py --foreground" >/dev/null; do sleep 5; done
: > /tmp/claude-1002/vvpre/exp_pitch.log
exec ~/.local/share/stickman-vibevoice/venv-rebuild/bin/python /tmp/claude-1002/vvpre/exp_pitch.py > /tmp/claude-1002/vvpre/exp_pitch.out 2>&1
