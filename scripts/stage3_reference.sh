#!/usr/bin/env bash
# Stage 3 reference row (B, m = 1) on CST val (addendum 5): inference, then the cache check. Direct run (addendum 4).
#   screen -dmS drone-s3r bash scripts/stage3_reference.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
LOG=results/stage3/logs
S="$LOG/stages.log"
# five minutes after the inference starts: per-frame time -> expected end, written to the log
( until grep -q "START r_cstval_b_m1" "$S"; do sleep 5; done; sleep 300
  line=$(grep "s/frame" "$LOG/r_cstval_b_m1.log" | tail -1)
  n=$(echo "$line" | sed -E 's/.*: ([0-9]+)\/([0-9]+) .*/\1/'); tot=$(echo "$line" | sed -E 's/.*: ([0-9]+)\/([0-9]+) .*/\2/'); spf=$(echo "$line" | sed -E 's/.* ([0-9.]+) s\/frame.*/\1/')
  eta=$(python3 -c "import time;print(time.strftime('%F %T', time.localtime(time.time()+($tot-$n)*$spf)))")
  echo "$(date '+%F %T %Z') ETA   r_cstval_b_m1: $n/$tot at $spf s/frame after 5 min -> expected end $eta (log line 300 s after start; frames since that line not counted)" | tee -a "$S" ) &
DIRECT=1 bash scripts/stage3_queue.sh scripts/queues3/06_reference_cst_val.txt; rc=$?
wait
echo "$(date '+%F %T %Z') REFERENCE END (exit $rc)" | tee -a "$S"
