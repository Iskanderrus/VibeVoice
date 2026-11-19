#!/usr/bin/env bash
# Podcast (RED/BLUE) per-scene generation, same guarded limits as the narration runs. Waits for foreign heavy jobs.
cd ~/Projects/tools/VibeVoice-pr8 || exit 1
V=~/.local/share/stickman-vibevoice/venv-rebuild
D=/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad/podcast
W=/tmp/claude-1002/vvpre/pod_scenes.log
T=$(date +%Y%m%d-%H%M%S)
: > /tmp/claude-1002/vvpre/pod_jobs.txt
for i in 1 2 3 4; do
  for try in 1 2 3 4 5 6; do
    while pgrep -f "graphifyy/bin/python|dartaotruntime|design_audit" >/dev/null; do sleep 2; done
    J=vv-pod-scene$i-$T; echo "$(date -u +%T) scene=$i try=$try start $J" >> $W
    $V/bin/python scripts/run_stickman_local.py --foreground --script $D/scene$i.txt --job-id $J > /tmp/claude-1002/vvpre/podscene$i.out 2>&1
    rc=$?; echo "$(date -u +%T) scene=$i try=$try exit=$rc" >> $W
    [ $rc -eq 0 ] && { echo "$J" >> /tmp/claude-1002/vvpre/pod_jobs.txt; break; }
    grep -q "PREFLIGHT" /tmp/claude-1002/vvpre/podscene$i.out || { echo "scene $i failed non-preflight, stop" >> $W; exit $rc; }
  done
done
echo ALLDONE >> $W
