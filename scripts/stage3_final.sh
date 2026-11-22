#!/usr/bin/env bash
# Stage 3 final inference in two lock groups (pre-registration 3, addendum 3):
#   group 1 = CST val (2 jobs), group 2 = Anti-UAV test (4 jobs). One heavy lock per group.
# Before each group: stop if memory pressure is warn or worse, or swap use is close to 2 GB (>= 1800 MB).
# After each group: one progress line in stages.log and the tools/heavy-status.sh output in heavy_status.log.
#   screen -dmS drone-s3f bash scripts/stage3_final.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/stage3/logs
S="$LOG/stages.log"
for g in 05a_final_cst_val 05b_final_antiuav; do
  level=$(sysctl -n kern.memorystatus_vm_pressure_level)      # 1 normal, 2 warn, 4 critical
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  if [ "$level" -ge 2 ] || [ "${swap%.*}" -ge 1800 ]; then
    echo "$(date '+%F %T %Z') HOLD  before $g: memory pressure level $level, swap used ${swap} MB - not started" | tee -a "$S"
    echo "$(date '+%F %T %Z') FINAL INFER STOPPED" | tee -a "$S"
    exit 3
  fi
  echo "$(date '+%F %T %Z') GROUP $g waiting for the heavy lock (pressure level $level, swap used ${swap} MB)" | tee -a "$S"
  ../../tools/heavy-run.sh env GROUP_LOCK=1 bash scripts/stage3_queue.sh "scripts/queues3/$g.txt"; rc=$?
  { echo "===== $(date '+%F %T %Z') after $g (exit $rc)"; bash ../../tools/heavy-status.sh 2>&1; } >> "$LOG/heavy_status.log"
  level=$(sysctl -n kern.memorystatus_vm_pressure_level)
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  echo "$(date '+%F %T %Z') GROUP $g ended (exit $rc): $(grep -c "DONE  " "$S") jobs done in total, pressure level $level, swap used ${swap} MB" | tee -a "$S"
  if [ $rc -ne 0 ]; then echo "$(date '+%F %T %Z') FINAL INFER STOPPED" | tee -a "$S"; exit $rc; fi
done
echo "$(date '+%F %T %Z') FINAL INFER END" | tee -a "$S"
