#!/usr/bin/env bash
# Wait for the first pitch comparison and any running VibeVoice job, then run the scene-2 (cfg 2.0, seed 18428) lowered-reference takes.
while pgrep -f "exp_pitch.py" >/dev/null || pgrep -f "run_stickman_local.py --foreground" >/dev/null; do sleep 5; done
: > /tmp/claude-1002/vvpre/exp_pitch2.log
exec ~/.local/share/stickman-vibevoice/venv-rebuild/bin/python /tmp/claude-1002/vvpre/exp_pitch2.py > /tmp/claude-1002/vvpre/exp_pitch2.out 2>&1
