#!/usr/bin/env bash
# Scene 2 (original dialogue), seed 18430, same recipe as the batch: Frank RED, Carter BLUE, guidance 1.6, boost off, guarded limits.
cd ~/Projects/tools/VibeVoice-pr8 || exit 1
V=~/.local/share/stickman-vibevoice/venv-rebuild
S=/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad/podcast/scene2.txt
W=/tmp/claude-1002/vvpre/s2_18430.log
for try in 1 2 3 4 5 6; do
  while pgrep -f "graphifyy/bin/python|dartaotruntime|design_audit" >/dev/null; do sleep 2; done
  J=vv-s2-seed18430-$(date +%Y%m%d-%H%M%S); echo "$(date +%T) try=$try start $J" >> $W
  $V/bin/python scripts/run_stickman_local.py --foreground --script $S --job-id $J --voice RED=$PWD/demo/voices/en-Frank_man.wav --voice BLUE=$PWD/demo/voices/en-Carter_man.wav --seed 18430 --cfg-scale 1.6 > /tmp/claude-1002/vvpre/s2_18430.out 2>&1
  rc=$?; echo "$(date +%T) try=$try exit=$rc" >> $W
  [ $rc -eq 0 ] && { echo DONE $J >> $W; exit 0; }
  grep -q PREFLIGHT /tmp/claude-1002/vvpre/s2_18430.out || exit $rc
done
