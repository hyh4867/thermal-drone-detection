#!/usr/bin/env bash
# Stage 3 job queue. Every job takes the shared heavy-work lock on its own (tools/heavy-run.sh),
# so heavy jobs of other projects can run between two jobs (pre-registration 3, addendum 1 item 3).
#   screen -dmS drone-s3 bash scripts/stage3_queue.sh scripts/queues3/<name>.txt
# Short jobs (selection, hold-out) may instead share one lock per queue file, so that each small job does not
# wait behind other projects' long jobs:   ../../tools/heavy-run.sh env GROUP_LOCK=1 bash scripts/stage3_queue.sh <file>
# The final inference always uses one lock per job (default mode).
# Queue file: one job per line, "label | python-module and arguments". Finished labels are
# skipped on restart; a failing job stops the queue unless its label starts with "?".
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=.venv/bin/python
HEAVY=../../tools/heavy-run.sh
Q=$1
LOG=results/stage3/logs
mkdir -p "$LOG"
S="$LOG/stages.log"
echo "$(date '+%F %T %Z') QUEUE $Q, commit $(git rev-parse --short HEAD), dirty $(git status --porcelain -- . ':(exclude)results' | wc -l | tr -d ' '), memory $(sysctl -n hw.memsize) B" | tee -a "$S"
while IFS='|' read -r label cmd; do
  label=$(echo "$label" | xargs); [ -z "$label" ] && continue
  case "$label" in \#*) continue;; esac
  optional=0; case "$label" in \?*) optional=1; label=${label#\?};; esac
  if grep -q "DONE  $label " "$S" 2>/dev/null; then echo "$(date '+%F %T %Z') SKIP  $label" | tee -a "$S"; continue; fi
  free=$(df -g . | awk 'NR==2{print $4}')
  if [ "$free" -lt 5 ]; then echo "$(date '+%F %T %Z') STOP  disk free ${free} GB < 5 GB" | tee -a "$S"; exit 2; fi
  t0=$SECONDS
  if [ "${DIRECT:-0}" = 1 ]; then
    # Addendum 4: the user released the heavy-run queue (2026-11-22 until the end of stage 3). Jobs start directly.
    # Before every job: log swap and memory pressure; do not start while swap use > 3 GB or free memory < 10%.
    while :; do
      sw=$(sysctl vm.swapusage); mp=$(memory_pressure | tail -1)
      echo "$(date '+%F %T %Z') CHECK before $label: $sw | $mp" | tee -a "$S"
      used=$(echo "$sw" | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); freep=$(echo "$mp" | sed -E 's/.*: ([0-9]+)%.*/\1/')
      if [ "${used%.*}" -le 3072 ] && [ "$freep" -ge 10 ]; then break; fi
      echo "$(date '+%F %T %Z') HOLD  $label: swap or free memory over the limit, checking again in 5 min" | tee -a "$S"
      sleep 300
    done
    echo "$(date '+%F %T %Z') START $label (direct)" | tee -a "$S"
    caffeinate -i $PY -m $cmd >> "$LOG/$label.log" 2>&1 < /dev/null; rc=$?
    if [ $rc -ne 0 ] && tail -30 "$LOG/$label.log" | grep -qiE "out of memory|MPS|Killed|MemoryError"; then
      echo "$(date '+%F %T %Z') RETRY $label once (exit $rc, memory / MPS error in the log), same settings" | tee -a "$S"
      caffeinate -i $PY -m $cmd >> "$LOG/$label.log" 2>&1 < /dev/null; rc=$?
    fi
  elif [ "${GROUP_LOCK:-0}" = 1 ]; then
    # the whole queue file already runs under ONE heavy lock (see below); no lock per job
    echo "$(date '+%F %T %Z') START $label (group lock)" | tee -a "$S"
    $PY -m $cmd >> "$LOG/$label.log" 2>&1 < /dev/null; rc=$?
  else
    echo "$(date '+%F %T %Z') WAIT  $label (heavy lock)" | tee -a "$S"
    "$HEAVY" $PY -m $cmd >> "$LOG/$label.log" 2>&1 < /dev/null; rc=$?
  fi
  if [ $rc = 0 ]; then
    echo "$(date '+%F %T %Z') DONE  $label ($(( (SECONDS - t0) / 60 )) min incl. lock wait)" | tee -a "$S"
  else
    echo "$(date '+%F %T %Z') FAIL  $label ($(( (SECONDS - t0) / 60 )) min, exit $rc) - see $LOG/$label.log" | tee -a "$S"
    [ $optional = 1 ] || exit 1
  fi
done < "$Q"
echo "$(date '+%F %T %Z') QUEUE DONE $Q" | tee -a "$S"
