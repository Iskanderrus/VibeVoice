#!/usr/bin/env bash
# Stop only the orchestrator processes named in $1 (a script basename); leave any in-flight synthesis take to finish.
for pid in $(pgrep -f "python .*/$1"); do kill "$pid" && echo "stopped orchestrator pid $pid"; done
