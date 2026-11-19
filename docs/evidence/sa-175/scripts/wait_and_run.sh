#!/usr/bin/env bash
# Wait for foreign heavy jobs (graphify, dart, design_audit) to finish, then start the guarded run at once.
cd ~/Projects/tools/VibeVoice-pr8 || exit 1
V=~/.local/share/stickman-vibevoice/venv-rebuild
S=/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad/blackswan_g1.txt
W=/tmp/claude-1002/vvpre/wait.log
for try in 1 2 3 4 5 6 7 8; do
  while pgrep -f "graphifyy/bin/python|dartaotruntime|design_audit" >/dev/null; do sleep 2; done
  J=vv-blackswan-g1c-$(date +%Y%m%d-%H%M%S); echo $J > /tmp/claude-1002/vvpre/job_full
  echo "$(date -u +%T) try=$try start $J" >> $W
  $V/bin/python scripts/run_stickman_local.py --foreground --script $S --job-id $J > /tmp/claude-1002/vvpre/full3.out 2>&1
  rc=$?; echo "$(date -u +%T) try=$try exit=$rc" >> $W
  [ $rc -eq 0 ] && exit 0
  grep -q "PREFLIGHT" /tmp/claude-1002/vvpre/full3.out || exit $rc
done
