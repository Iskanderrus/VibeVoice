#!/bin/bash
# Separate-process watchdog: pause the actual synthesis worker, resume after cooling.
set -euo pipefail
PID="$1"; LOG="$2"
GPU_PAUSE=${THERMAL_GPU_PAUSE_C:-75}
CPU_PAUSE=${THERMAL_CPU_PAUSE_C:-80}
GPU_RESUME=${THERMAL_GPU_RESUME_C:-65}
CPU_RESUME=${THERMAL_CPU_RESUME_C:-70}
CPU_CRITICAL=90
STOPPED=0
while kill -0 "$PID" 2>/dev/null; do
  G="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -cd '0-9' || true)"
  C="$(sensors 2>/dev/null | awk '/Tctl:|Package id 0:|Core 0:/{for(i=1;i<=NF;i++)if($i~/\+[0-9]+(\.[0-9]+)?/){gsub(/[^0-9.]/,"",$i);print int($i);exit}}' || true)"
  echo "$(date -u +%FT%TZ) gpu=${G:-na} cpu=${C:-na} stopped=$STOPPED" >> "$LOG"
  if [ -z "$C" ] || [ -z "$G" ]; then
    echo "BLOCKED: required CPU/GPU thermal sensor unavailable" >> "$LOG"
    kill -CONT "$PID" 2>/dev/null || true
    kill -TERM "$PID" 2>/dev/null || true
    exit 2
  fi
  if [ "$C" -ge "$CPU_CRITICAL" ]; then
    echo "CRITICAL_THERMAL_STOP cpu=$C" >> "$LOG"
    kill -CONT "$PID" 2>/dev/null || true
    kill -TERM "$PID" 2>/dev/null || true
    exit 3
  fi
  if [ "$STOPPED" = 0 ] && { [ "$G" -ge "$GPU_PAUSE" ] || [ "$C" -ge "$CPU_PAUSE" ]; }; then
    kill -STOP "$PID" 2>/dev/null || true
    STOPPED=1
    echo "PAUSE" >> "$LOG"
  elif [ "$STOPPED" = 1 ] && [ "$G" -le "$GPU_RESUME" ] && [ "$C" -le "$CPU_RESUME" ]; then
    kill -CONT "$PID" 2>/dev/null || true
    STOPPED=0
    echo "RESUME" >> "$LOG"
  fi
  sleep 1
done
echo "process exited" >> "$LOG"
