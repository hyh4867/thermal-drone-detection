#!/usr/bin/env bash
# Stage 2 job queue: runs the jobs of a queue file one at a time (8 GB machine).
#   caffeinate -i bash scripts/stage2_queue.sh scripts/queues/<name>.txt
# Queue file: one job per line, "label | python-module and arguments". Finished
# labels are skipped on restart; a failing job is logged and the queue continues
# only if the line starts with "?" (optional job), otherwise it stops.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=.venv/bin/python
Q=$1
LOG=results/stage2/logs
mkdir -p "$LOG"
S="$LOG/stages.log"
echo "$(date '+%F %T %Z') QUEUE $Q, commit $(git rev-parse --short HEAD), memory $(sysctl -n hw.memsize) B" | tee -a "$S"
while IFS='|' read -r label cmd; do
  label=$(echo "$label" | xargs); [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  optional=0; case "$label" in \?*) optional=1; label=${label#\?};; esac
  if grep -q "DONE  $label " "$S" 2>/dev/null; then echo "$(date '+%F %T %Z') SKIP  $label" | tee -a "$S"; continue; fi
  echo "$(date '+%F %T %Z') START $label" | tee -a "$S"
  t0=$SECONDS
  if $PY -m $cmd >> "$LOG/$label.log" 2>&1; then
    echo "$(date '+%F %T %Z') DONE  $label ($(( (SECONDS - t0) / 60 )) min)" | tee -a "$S"
  else
    echo "$(date '+%F %T %Z') FAIL  $label ($(( (SECONDS - t0) / 60 )) min, exit $?) - see $LOG/$label.log" | tee -a "$S"
    [ $optional = 1 ] || exit 1
  fi
done < "$Q"
echo "$(date '+%F %T %Z') QUEUE DONE $Q" | tee -a "$S"
