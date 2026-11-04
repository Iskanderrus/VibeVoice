#!/bin/bash
# Shared thermal readers for the hybrid runner. Each prints an integer in °C, or nothing
# when the sensor is unavailable. Sourced by thermal_preflight.sh and thermal_guard.sh.

read_cpu_c() {
  sensors 2>/dev/null | awk '/Tctl:|Package id 0:|Core 0:/{for(i=1;i<=NF;i++)if($i~/\+[0-9]+(\.[0-9]+)?/){gsub(/[^0-9.]/,"",$i);print int($i);exit}}' || true
}

read_gpu_c() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -cd '0-9' || true
}
