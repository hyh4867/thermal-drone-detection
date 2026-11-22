#!/usr/bin/env bash
# Direct run (addendum 4) of a stage 3 queue file, with an expected-end line 5 minutes after each inference job starts.
#   screen -dmS drone-s3x bash scripts/stage3_direct.sh scripts/queues3/<file>.txt
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
LOG=results/stage3/logs
S="$LOG/stages.log"
Q=$1
for label in $(grep "tdd.stage3.infer" "$Q" | cut -d'|' -f1 | tr -d ' ?'); do
  ( until grep -q "START $label (direct)" "$S"; do sleep 5; done; sleep 300
    line=$(grep "s/frame" "$LOG/$label.log" 2>/dev/null | tail -1)
    [ -z "$line" ] && exit 0
    n=$(echo "$line" | sed -E 's/.*: ([0-9]+)\/([0-9]+) .*/\1/'); tot=$(echo "$line" | sed -E 's/.*: ([0-9]+)\/([0-9]+) .*/\2/'); spf=$(echo "$line" | sed -E 's/.* ([0-9.]+) s\/frame.*/\1/')
    t_line=$(stat -f %m "$LOG/$label.log")
    eta=$(python3 -c "import time;print(time.strftime('%F %T', time.localtime($t_line+($tot-$n)*$spf)))")
    echo "$(date '+%F %T %Z') ETA   $label: $n/$tot at $spf s/frame, 5 min after start -> expected end $eta" >> "$S" ) &
done
DIRECT=1 bash scripts/stage3_queue.sh "$Q"; rc=$?
wait
echo "$(date '+%F %T %Z') DIRECT END $Q (exit $rc)" | tee -a "$S"
