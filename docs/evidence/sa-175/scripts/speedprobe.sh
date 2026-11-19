#!/usr/bin/env bash
# Model-free speed/thermal probe: more cores/threads, boost off. Self-stops at raw CPU>=85 or GPU>=72.
source ~/Projects/tools/VibeVoice-pr8/scripts/local-hybrid/thermal_sensors.sh
V=~/.local/share/stickman-vibevoice/venv-rebuild
OUT=/tmp/claude-1002/vvpre/speed.log
DUR=${DUR:-120}
: > $OUT
cat > /tmp/claude-1002/vvpre/burn2.py <<'P'
import os, sys, time, torch
n = int(os.environ["BURN_THREADS"]); out = os.environ["BURN_OUT"]
torch.set_num_threads(n)
a = torch.randn(1536, 1536); b = torch.randn(1536, 1536)
it = 0; t0 = time.time()
while True:
    a = (a @ b).tanh(); it += 1
    if it % 5 == 0:
        open(out, "w").write(f"{it/(time.time()-t0):.3f}")
P
median() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
cool() {  # CPU<=62 and GPU<=60 for 6 consecutive samples, max 5 min
  local n=0 t=0 c g
  while [ $t -lt 600 ]; do
    c=$(read_cpu_c); g=$(read_gpu_c); t=$((t+1)); sleep 0.5
    if [ -n "$c" ] && [ -n "$g" ] && [ "$c" -le 62 ] && [ "$g" -le 60 ]; then n=$((n+1)); else n=0; fi
    [ $n -ge 6 ] && return 0
  done
  return 1
}
# label threads cpus quota
for cfg in "2t 2 0,2 150" "4t 4 0-3 300" "6t 6 0-5 450" "8t 8 0-7 600"; do
  set -- $cfg; L=$1; TH=$2; CPUS=$3; Q=$4
  if ! cool; then echo "$L COOLDOWN_FAILED (machine not cool; other load?)" >> $OUT; break; fi
  U=vv-speed-$L-$$; RF=/tmp/claude-1002/vvpre/burn2.rate; rm -f $RF
  BURN_THREADS=$TH BURN_OUT=$RF OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH \
    systemd-run --user --scope --quiet --unit=$U -p CPUQuota=${Q}% taskset -c $CPUS $V/bin/python /tmp/claude-1002/vvpre/burn2.py >/dev/null 2>&1 &
  sleep 1
  cs=(); gs=(); abort=0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $DUR ]; do
    c=$(read_cpu_c); g=$(read_gpu_c)
    if [ -z "$c" ] || [ -z "$g" ]; then abort=2; break; fi
    cs+=("$c"); gs+=("$g")
    if [ "$c" -ge 85 ]; then abort=1; break; fi
    if [ "$g" -ge 72 ]; then abort=3; break; fi
    sleep 0.5
  done
  thr=$(cat $RF 2>/dev/null || echo 0)
  systemctl --user stop $U.scope 2>/dev/null; pkill -f /tmp/claude-1002/vvpre/burn2.py 2>/dev/null
  wait 2>/dev/null
  n=${#cs[@]}; cmax=$(printf '%s\n' "${cs[@]}" | sort -n | tail -1); cmed=$(printf '%s\n' "${cs[@]}" | median)
  gmax=$(printf '%s\n' "${gs[@]}" | sort -n | tail -1)
  ge80=$(printf '%s\n' "${cs[@]}" | awk '$1>=80' | wc -l); ge75=$(printf '%s\n' "${cs[@]}" | awk '$1>=75' | wc -l)
  echo "$L threads=$TH cpus=$CPUS quota=${Q}% n=$n secs=$(( $(date +%s) - t0 )) cpu_max=$cmax cpu_med=$cmed ge75=$ge75 ge80=$ge80 gpu_max=$gmax matmul_per_s=$thr abort=$abort" >> $OUT
  [ $abort -ne 0 ] && break
  [ "$ge80" -gt 0 ] && break
done
echo DONE >> $OUT
