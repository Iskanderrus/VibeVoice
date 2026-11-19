#!/usr/bin/env bash
# Wait for any running VibeVoice job (the v2 take already in flight) to exit, then start the v3 orchestrator.
while pgrep -f "run_stickman_local.py --foreground" >/dev/null; do sleep 3; done
: > /tmp/claude-1002/vvpre/pod3.log
exec ~/.local/share/stickman-vibevoice/venv-rebuild/bin/python /tmp/claude-1002/vvpre/pod_v3.py > /tmp/claude-1002/vvpre/pod3.out 2>&1
