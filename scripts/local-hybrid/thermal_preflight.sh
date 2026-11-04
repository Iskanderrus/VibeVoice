#!/bin/bash
# Gate run before any model load: start only when CPU and GPU are at or below the
# resume limits. Waits a bounded time for cooling, then refuses to start and records why.
# Usage: thermal_preflight.sh <thermal.log>   (exit 0 = safe to start, 2 = blocked)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/thermal_sensors.sh"
LOG="$1"
START_CPU_MAX=${THERMAL_CPU_RESUME_C:-70}
START_GPU_MAX=${THERMAL_GPU_RESUME_C:-65}
WAIT_MAX=${THERMAL_PREFLIGHT_WAIT_SECONDS:-300}
POLL=${THERMAL_PREFLIGHT_POLL_SECONDS:-5}
START=$(date +%s)
while :; do
  G="$(read_gpu_c)"
  C="$(read_cpu_c)"
  if [ -z "$C" ] || [ -z "$G" ]; then
    echo "BLOCKED: required CPU/GPU thermal sensor unavailable before start" >> "$LOG"
    exit 2
  fi
  if [ "$C" -le "$START_CPU_MAX" ] && [ "$G" -le "$START_GPU_MAX" ]; then
    echo "$(date -u +%FT%TZ) PREFLIGHT_OK gpu=$G cpu=$C" >> "$LOG"
    exit 0
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -ge "$WAIT_MAX" ]; then
    echo "BLOCKED: thermal preflight not met after ${ELAPSED}s gpu=$G cpu=$C (need cpu<=$START_CPU_MAX gpu<=$START_GPU_MAX)" >> "$LOG"
    exit 2
  fi
  echo "$(date -u +%FT%TZ) PREFLIGHT_WAIT gpu=$G cpu=$C" >> "$LOG"
  sleep "$POLL"
done
