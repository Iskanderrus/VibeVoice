#!/bin/bash
# Separate-process watchdog for one synthesis process tree (root PID plus descendants).
# Pauses the tree when hot, resumes only after cooling, and terminates it with a
# recorded reason if cooling does not happen within THERMAL_PAUSE_MAX_SECONDS.
# Usage: thermal_guard.sh <pid> <thermal.log>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/thermal_sensors.sh"
PID="$1"; LOG="$2"
GPU_PAUSE=${THERMAL_GPU_PAUSE_C:-75}
CPU_PAUSE=${THERMAL_CPU_PAUSE_C:-80}
GPU_RESUME=${THERMAL_GPU_RESUME_C:-65}
CPU_RESUME=${THERMAL_CPU_RESUME_C:-70}
PAUSE_MAX=${THERMAL_PAUSE_MAX_SECONDS:-240}
CPU_CRITICAL=90
STOPPED=0
PAUSED_AT=0

tree_pids() {  # root first, then every descendant
  local pid=$1 child
  echo "$pid"
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do tree_pids "$child"; done
}
signal_tree() {  # signal_tree <SIGNAL> <root pid>
  local pid
  for pid in $(tree_pids "$2"); do kill "-$1" "$pid" 2>/dev/null || true; done
}
stop_worker() {  # resume first so the workers can receive TERM, then terminate the tree
  signal_tree CONT "$PID"
  signal_tree TERM "$PID"
}

while kill -0 "$PID" 2>/dev/null; do
  G="$(read_gpu_c)"
  C="$(read_cpu_c)"
  echo "$(date -u +%FT%TZ) gpu=${G:-na} cpu=${C:-na} stopped=$STOPPED" >> "$LOG"
  if [ -z "$C" ] || [ -z "$G" ]; then
    echo "BLOCKED: required CPU/GPU thermal sensor unavailable" >> "$LOG"
    stop_worker
    exit 2
  fi
  if [ "$C" -ge "$CPU_CRITICAL" ]; then
    echo "CRITICAL_THERMAL_STOP cpu=$C" >> "$LOG"
    stop_worker
    exit 3
  fi
  NOW=$(date +%s)
  if [ "$STOPPED" = 1 ] && [ $((NOW - PAUSED_AT)) -ge "$PAUSE_MAX" ]; then
    echo "THERMAL_PAUSE_TIMEOUT cpu=$C gpu=$G paused_seconds=$((NOW - PAUSED_AT)) limit=$PAUSE_MAX" >> "$LOG"
    stop_worker
    exit 4
  fi
  if [ "$STOPPED" = 0 ] && { [ "$G" -ge "$GPU_PAUSE" ] || [ "$C" -ge "$CPU_PAUSE" ]; }; then
    signal_tree STOP "$PID"
    STOPPED=1
    PAUSED_AT=$NOW
    echo "PAUSE" >> "$LOG"
  elif [ "$STOPPED" = 1 ] && [ "$G" -le "$GPU_RESUME" ] && [ "$C" -le "$CPU_RESUME" ]; then
    signal_tree CONT "$PID"
    STOPPED=0
    echo "RESUME" >> "$LOG"
  fi
  sleep 1
done
echo "process exited" >> "$LOG"
