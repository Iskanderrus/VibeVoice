#!/usr/bin/env bash
# Fix 1: one guarded run per scene (same limits). Waits for foreign heavy jobs; retries on preflight block.
cd ~/Projects/tools/VibeVoice-pr8 || exit 1
V=~/.local/share/stickman-vibevoice/venv-rebuild
D=/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad
W=/tmp/claude-1002/vvpre/scenes.log
T=$(date +%Y%m%d-%H%M%S)
i=0
while IFS= read -r line; do
  i=$((i+1)); S=$D/scene$i.txt; printf '%s\n' "$line" > $S
  for try in 1 2 3 4 5 6; do
    while pgrep -f "graphifyy/bin/python|dartaotruntime|design_audit" >/dev/null; do sleep 2; done
    J=vv-bs-scene$i-$T; echo "$(date -u +%T) scene=$i try=$try start $J" >> $W
    $V/bin/python scripts/run_stickman_local.py --foreground --script $S --job-id $J > /tmp/claude-1002/vvpre/scene$i.out 2>&1
    rc=$?; echo "$(date -u +%T) scene=$i try=$try exit=$rc" >> $W
    [ $rc -eq 0 ] && { echo "$J" >> /tmp/claude-1002/vvpre/scene_jobs.txt; break; }
    grep -q "PREFLIGHT" /tmp/claude-1002/vvpre/scene$i.out || { echo "scene $i failed non-preflight, stop" >> $W; exit $rc; }
  done
done < $D/blackswan_g1.txt
echo ALLDONE >> $W
