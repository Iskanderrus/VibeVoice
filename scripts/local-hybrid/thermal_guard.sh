#!/bin/bash
# Pause (SIGSTOP) a process while GPU or CPU is hot; resume (SIGCONT) when cool.
# usage: thermal_guard.sh <pid> <log>
PID="$1"; LOG="$2"
GPU_PAUSE=${THERMAL_GPU_PAUSE_C:-80}; CPU_PAUSE=${THERMAL_CPU_PAUSE_C:-88}
GPU_RESUME=${THERMAL_GPU_RESUME_C:-68}; CPU_RESUME=${THERMAL_CPU_RESUME_C:-78}
STOPPED=0
while kill -0 "$PID" 2>/dev/null; do
  G=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -d ' ')
  C=$(sensors 2>/dev/null | awk '/Tctl|Package id 0|Core 0/{gsub(/[+°C]/,"",$2); print int($2); exit}')
  echo "$(date -u +%T) gpu=$G cpu=${C:-na} stopped=$STOPPED" >> "$LOG"
  if [ "$STOPPED" = 0 ] && { [ "$G" -ge "$GPU_PAUSE" ] || [ "${C:-0}" -ge "$CPU_PAUSE" ]; }; then
    kill -STOP "$PID"; STOPPED=1; echo "PAUSE" >> "$LOG"
  fi
  if [ "$STOPPED" = 1 ] && [ "$G" -le "$GPU_RESUME" ] && [ "${C:-0}" -le "$CPU_RESUME" ]; then
    kill -CONT "$PID"; STOPPED=0; echo "RESUME" >> "$LOG"
  fi
  sleep 5
done
echo "process exited" >> "$LOG"
