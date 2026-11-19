#!/bin/bash
# Gate run before any model load: start only from a sustained, measured safe idle. A single cool
# reading is not enough: THERMAL_PREFLIGHT_STABLE_SAMPLES consecutive samples must all be at or
# below the resume limits, the window median must sit THERMAL_PREFLIGHT_MEDIAN_MARGIN_C below them,
# and the second half of the window may not run hotter than the first by more than
# THERMAL_PREFLIGHT_MAX_TREND_C (no rising trend; single-sample Tctl noise is not a trend), and the CPU must be at least THERMAL_PREFLIGHT_MIN_IDLE_PERCENT idle over that same window
# (invalid or stale counters fail closed), with THERMAL_MIN_AVAILABLE_MEM_MIB of RAM available, so
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
MARGIN=${THERMAL_PREFLIGHT_MEDIAN_MARGIN_C:-5}
TREND=${THERMAL_PREFLIGHT_MAX_TREND_C:-2}
MIN_IDLE=${THERMAL_PREFLIGHT_MIN_IDLE_PERCENT:-70}
MIN_MEM_MIB=${THERMAL_MIN_AVAILABLE_MEM_MIB:-12288}
START=$(date +%s)
CPU_WIN=(); GPU_WIN=(); IDLE_WIN=(); TOTAL_WIN=()

cpu_times() {  # prints "<idle+iowait> <total>"; prints nothing when the counters are unusable
  awk '/^cpu /{ if (NF < 6) exit; i=$5+$6; t=0; ok=1
    for(n=2;n<=NF;n++){ if ($n !~ /^[0-9]+$/) ok=0; t+=$n }
    if (ok && t > 0 && i <= t) print i, t; exit }' "${THERMAL_PROC_STAT:-/proc/stat}" 2>/dev/null || true
}
mem_available_mib() {  # MemAvailable in MiB, or nothing when unreadable
  awk '/^MemAvailable:/{ if ($2 ~ /^[0-9]+$/) print int($2/1024); exit }' "${THERMAL_PROC_MEMINFO:-/proc/meminfo}" 2>/dev/null || true
}
median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
trend() {  # median of the second half minus median of the first half
  local n=$# half; half=$(( $# / 2 ))
  local first=("${@:1:half}") last=("${@:$((n - half + 1)):half}")
  echo $(( $(median "${last[@]}") - $(median "${first[@]}") ))
}
minmax() { printf '%s\n' "$@" | sort -n | awk 'NR==1{lo=$1}{hi=$1}END{print lo, hi}'; }
drop_first() { CPU_WIN=("${CPU_WIN[@]:1}"); GPU_WIN=("${GPU_WIN[@]:1}"); IDLE_WIN=("${IDLE_WIN[@]:1}"); TOTAL_WIN=("${TOTAL_WIN[@]:1}"); }
reset_window() { CPU_WIN=(); GPU_WIN=(); IDLE_WIN=(); TOTAL_WIN=(); }

while :; do
  G="$(read_gpu_c)"
  C="$(read_cpu_c)"
  if [ -z "$C" ] || [ -z "$G" ]; then
    echo "BLOCKED: required CPU/GPU thermal sensor unavailable before start" >> "$LOG"
    exit 2
  fi
  read -r IDLE_NOW TOTAL_NOW <<< "$(cpu_times)"
  if [ -z "${IDLE_NOW:-}" ] || [ -z "${TOTAL_NOW:-}" ]; then
    echo "BLOCKED: CPU idle counters unreadable or invalid before start" >> "$LOG"
    exit 2
  fi
  if [ "$C" -le "$START_CPU_MAX" ] && [ "$G" -le "$START_GPU_MAX" ]; then
    CPU_WIN+=("$C"); GPU_WIN+=("$G"); IDLE_WIN+=("$IDLE_NOW"); TOTAL_WIN+=("$TOTAL_NOW")
  else
    reset_window  # any hot sample restarts the stable window
  fi
  if [ "${#CPU_WIN[@]}" -ge "$STABLE" ]; then
    # Idle is measured over exactly the retained stable window: first to last sample counters.
    DT=$((TOTAL_WIN[${#TOTAL_WIN[@]} - 1] - TOTAL_WIN[0]))
    DI=$((IDLE_WIN[${#IDLE_WIN[@]} - 1] - IDLE_WIN[0]))
    MEM="$(mem_available_mib)"
    read -r CLO CHI <<< "$(minmax "${CPU_WIN[@]}")"
    read -r GLO GHI <<< "$(minmax "${GPU_WIN[@]}")"
    if [ "$DT" -le 0 ] || [ "$DI" -lt 0 ] || [ "$DI" -gt "$DT" ]; then
      echo "$(date -u +%FT%TZ) PREFLIGHT_INVALID_IDLE_COUNTERS delta_total=$DT delta_idle=$DI (stale or non-monotonic)" >> "$LOG"
      drop_first
    elif [ -z "$MEM" ]; then
      echo "BLOCKED: MemAvailable unreadable before start" >> "$LOG"
      exit 2
    else
      IDLE_PCT=$(( 100 * DI / DT ))
      CMED="$(median "${CPU_WIN[@]}")"; GMED="$(median "${GPU_WIN[@]}")"
      CTR="$(trend "${CPU_WIN[@]}")"; GTR="$(trend "${GPU_WIN[@]}")"
      if [ "$CMED" -le $((START_CPU_MAX - MARGIN)) ] && [ "$GMED" -le $((START_GPU_MAX - MARGIN)) ] \
         && [ "$CTR" -le "$TREND" ] && [ "$GTR" -le "$TREND" ] \
         && [ "$IDLE_PCT" -ge "$MIN_IDLE" ] && [ "$MEM" -ge "$MIN_MEM_MIB" ]; then
        echo "$(date -u +%FT%TZ) PREFLIGHT_OK gpu=$G cpu=$C window_cpu=${CLO}-${CHI} med=$CMED trend=$CTR window_gpu=${GLO}-${GHI} med=$GMED trend=$GTR idle_pct=$IDLE_PCT mem_available_mib=$MEM samples=${#CPU_WIN[@]}" >> "$LOG"
        exit 0
      fi
      echo "$(date -u +%FT%TZ) PREFLIGHT_UNSTABLE cpu=${CLO}-${CHI} med=$CMED trend=$CTR gpu=${GLO}-${GHI} med=$GMED trend=$GTR idle_pct=$IDLE_PCT mem_available_mib=$MEM (need median<=$((START_CPU_MAX - MARGIN))/$((START_GPU_MAX - MARGIN)) trend<=$TREND idle>=$MIN_IDLE mem>=$MIN_MEM_MIB)" >> "$LOG"
      drop_first
    fi
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -ge "$WAIT_MAX" ]; then
    echo "BLOCKED: thermal preflight not met after ${ELAPSED}s gpu=$G cpu=$C (need ${STABLE} stable samples cpu<=$START_CPU_MAX gpu<=$START_GPU_MAX median margin ${MARGIN} trend<=$TREND idle>=${MIN_IDLE}% mem>=${MIN_MEM_MIB}MiB)" >> "$LOG"
    exit 2
  fi
  echo "$(date -u +%FT%TZ) PREFLIGHT_WAIT gpu=$G cpu=$C stable=${#CPU_WIN[@]}/$STABLE" >> "$LOG"
  sleep "$POLL"
done
