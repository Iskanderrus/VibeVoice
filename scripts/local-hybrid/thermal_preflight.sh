#!/bin/bash
# Gate run before any model load: start only from a sustained, measured safe idle. A single cool
# reading is not enough: THERMAL_PREFLIGHT_STABLE_SAMPLES consecutive samples must all be at or
# below the resume limits, stay within THERMAL_PREFLIGHT_MAX_SPREAD_C of each other (no rising
# trend), and the CPU must be at least THERMAL_PREFLIGHT_MIN_IDLE_PERCENT idle over the window so
# unrelated load cannot push the machine over the limit once the model starts. Waits a bounded
# time, then refuses to start and records why. Never kills or alters other processes.
# Usage: thermal_preflight.sh <thermal.log>   (exit 0 = safe to start, 2 = blocked)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/thermal_sensors.sh"
LOG="$1"
START_CPU_MAX=${THERMAL_CPU_RESUME_C:-70}
START_GPU_MAX=${THERMAL_GPU_RESUME_C:-65}
WAIT_MAX=${THERMAL_PREFLIGHT_WAIT_SECONDS:-300}
POLL=${THERMAL_PREFLIGHT_POLL_SECONDS:-5}
STABLE=${THERMAL_PREFLIGHT_STABLE_SAMPLES:-6}
SPREAD=${THERMAL_PREFLIGHT_MAX_SPREAD_C:-4}
MIN_IDLE=${THERMAL_PREFLIGHT_MIN_IDLE_PERCENT:-70}
START=$(date +%s)
CPU_WIN=(); GPU_WIN=()

cpu_times() {  # prints "<idle+iowait> <total>" from the aggregate /proc/stat line
  awk '/^cpu /{i=$5+$6; t=0; for(n=2;n<=NF;n++)t+=$n; print i, t; exit}' "${THERMAL_PROC_STAT:-/proc/stat}"
}
read -r IDLE0 TOTAL0 <<< "$(cpu_times)"

minmax() { printf '%s\n' "$@" | sort -n | awk 'NR==1{lo=$1}{hi=$1}END{print lo, hi}'; }

while :; do
  G="$(read_gpu_c)"
  C="$(read_cpu_c)"
  if [ -z "$C" ] || [ -z "$G" ]; then
    echo "BLOCKED: required CPU/GPU thermal sensor unavailable before start" >> "$LOG"
    exit 2
  fi
  if [ "$C" -le "$START_CPU_MAX" ] && [ "$G" -le "$START_GPU_MAX" ]; then
    CPU_WIN+=("$C"); GPU_WIN+=("$G")
  else
    CPU_WIN=(); GPU_WIN=()  # any hot sample restarts the stable window
    read -r IDLE0 TOTAL0 <<< "$(cpu_times)"
  fi
  if [ "${#CPU_WIN[@]}" -ge "$STABLE" ]; then
    read -r IDLE1 TOTAL1 <<< "$(cpu_times)"
    DT=$((TOTAL1 - TOTAL0))
    IDLE_PCT=100
    if [ "$DT" -gt 0 ]; then IDLE_PCT=$(( 100 * (IDLE1 - IDLE0) / DT )); fi
    read -r CLO CHI <<< "$(minmax "${CPU_WIN[@]}")"
    read -r GLO GHI <<< "$(minmax "${GPU_WIN[@]}")"
    if [ $((CHI - CLO)) -le "$SPREAD" ] && [ $((GHI - GLO)) -le "$SPREAD" ] && [ "$IDLE_PCT" -ge "$MIN_IDLE" ]; then
      echo "$(date -u +%FT%TZ) PREFLIGHT_OK gpu=$G cpu=$C window_cpu=${CLO}-${CHI} window_gpu=${GLO}-${GHI} idle_pct=$IDLE_PCT samples=${#CPU_WIN[@]}" >> "$LOG"
      exit 0
    fi
    echo "$(date -u +%FT%TZ) PREFLIGHT_UNSTABLE cpu=${CLO}-${CHI} gpu=${GLO}-${GHI} idle_pct=$IDLE_PCT (need spread<=$SPREAD idle>=$MIN_IDLE)" >> "$LOG"
    CPU_WIN=("${CPU_WIN[@]:1}"); GPU_WIN=("${GPU_WIN[@]:1}")
    read -r IDLE0 TOTAL0 <<< "$(cpu_times)"
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -ge "$WAIT_MAX" ]; then
    echo "BLOCKED: thermal preflight not met after ${ELAPSED}s gpu=$G cpu=$C (need ${STABLE} stable samples cpu<=$START_CPU_MAX gpu<=$START_GPU_MAX spread<=$SPREAD idle>=${MIN_IDLE}%)" >> "$LOG"
    exit 2
  fi
  echo "$(date -u +%FT%TZ) PREFLIGHT_WAIT gpu=$G cpu=$C stable=${#CPU_WIN[@]}/$STABLE" >> "$LOG"
  sleep "$POLL"
done
