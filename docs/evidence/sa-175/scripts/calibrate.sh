#!/usr/bin/env bash
# Model-free quota calibration: fp32 matmul burner (like the CPU language model) under the same scope/affinity as the runner.
source ~/Projects/tools/VibeVoice-pr8/scripts/local-hybrid/thermal_sensors.sh
V=~/.local/share/stickman-vibevoice/venv-rebuild
OUT=/tmp/claude-1002/vvpre/calib.log
DUR=${DUR:-120}
: > $OUT
cat > /tmp/claude-1002/vvpre/burn.py <<'P'
import os, time, torch
torch.set_num_threads(2)
a = torch.randn(1536, 1536); b = torch.randn(1536, 1536)
while True:
    a = (a @ b).tanh()
P
median() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
cool() {  # wait until 6 consecutive raw samples <= 62, max 300 s
  local n=0 t=0 c
  while [ $t -lt 600 ]; do
    c=$(read_cpu_c); t=$((t+1)); sleep 0.5
    if [ -n "$c" ] && [ "$c" -le 62 ]; then n=$((n+1)); else n=0; fi
    [ $n -ge 6 ] && return 0
  done
  return 1
}
for Q in 50 75 100 125; do
  if ! cool; then echo "Q=$Q COOLDOWN_FAILED (machine not cool; other load?)" >> $OUT; break; fi
  U=vv-calib-$Q-$$
  systemd-run --user --scope --quiet --unit=$U -p CPUQuota=${Q}% taskset -c 0,2 $V/bin/python /tmp/claude-1002/vvpre/burn.py >/dev/null 2>&1 &
  sleep 1
  samples=(); abort=0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $DUR ]; do
    c=$(read_cpu_c); g=$(read_gpu_c)
    if [ -z "$c" ]; then abort=2; break; fi
    samples+=("$c")
    if [ "$c" -ge 85 ]; then abort=1; break; fi
    sleep 0.5
  done
  systemctl --user stop $U.scope 2>/dev/null; pkill -f /tmp/claude-1002/vvpre/burn.py 2>/dev/null
  wait 2>/dev/null
  n=${#samples[@]}; max=$(printf '%s\n' "${samples[@]}" | sort -n | tail -1)
  med=$(printf '%s\n' "${samples[@]}" | median)
  ge80=$(printf '%s\n' "${samples[@]}" | awk '$1>=80' | wc -l)
  ge75=$(printf '%s\n' "${samples[@]}" | awk '$1>=75' | wc -l)
  echo "Q=${Q}% n=$n secs=$(( $(date +%s) - t0 )) max=$max median=$med ge75=$ge75 ge80=$ge80 abort=$abort" >> $OUT
  [ $abort -ne 0 ] && break
  [ "$ge80" -gt 0 ] && break
done
echo DONE >> $OUT
