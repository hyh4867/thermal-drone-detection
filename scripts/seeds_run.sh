#!/usr/bin/env bash
# Seed study of T2-20 (docs/PREREGISTRATION_SEEDS.md): seeds 1 and 2, one after the other, unattended.
#   screen -dmS drone-seeds bash scripts/seeds_run.sh
# Per seed: training -> hold-out inference -> cache check -> freeze -> COMMIT of the frozen file -> CST val ->
# Anti-UAV test (+ supplement) -> Svanström test -> cache check -> evaluation -> commit. Jobs start directly (no heavy
# lock, as instructed for this night). Before every job: swap and free memory are logged; no start while swap use is
# over 3 GB or free memory is below 10% (checked again every 5 minutes). Finished labels are skipped on restart.
# Time limits (KST): seed 2 training starts only before TRAIN2_BY; no long job starts after LONG_BY.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=.venv/bin/python
LOG=results/seeds/logs
mkdir -p "$LOG"
S="$LOG/stages.log"
TRAIN2_BY=$(date -j -f '%F %T' "${TRAIN2_BY:-2026-11-24 02:30:00}" +%s)
LONG_BY=$(date -j -f '%F %T' "${LONG_BY:-2026-11-24 04:30:00}" +%s)
say() { echo "$(date '+%F %T %Z') $*" | tee -a "$S"; }
finish() { say "RUN END (exit $1): $2"; exit "$1"; }

job() {   # job <label> <long: 0|1> <python module and arguments>
  local label=$1 long=$2; shift 2
  if grep -q "DONE  $label " "$S" 2>/dev/null; then say "SKIP  $label"; return 0; fi
  while :; do
    if [ "$long" = 1 ] && [ "$(date +%s)" -ge "$LONG_BY" ]; then finish 4 "$label not started: after the limit for long jobs"; fi
    local sw mp used freep
    sw=$(sysctl vm.swapusage); mp=$(memory_pressure | tail -1)
    say "CHECK before $label: $sw | $mp"
    used=$(echo "$sw" | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); freep=$(echo "$mp" | sed -E 's/.*: ([0-9]+)%.*/\1/')
    if [ "${used%.*}" -le 3072 ] && [ "$freep" -ge 10 ]; then break; fi
    say "HOLD  $label: swap or free memory over the limit, checking again in 5 min"
    sleep 300
  done
  local t0=$SECONDS rc
  say "START $label (commit $(git rev-parse --short HEAD))"
  caffeinate -i $PY -m "$@" >> "$LOG/$label.log" 2>&1 < /dev/null; rc=$?
  if [ $rc -ne 0 ] && tail -30 "$LOG/$label.log" | grep -qiE "out of memory|MPS|Killed|MemoryError"; then
    say "RETRY $label once (exit $rc, memory / MPS error in the log), same settings"
    caffeinate -i $PY -m "$@" >> "$LOG/$label.log" 2>&1 < /dev/null; rc=$?
  fi
  if [ $rc = 0 ]; then say "DONE  $label ($(( (SECONDS - t0) / 60 )) min)"; return 0; fi
  say "FAIL  $label ($(( (SECONDS - t0) / 60 )) min, exit $rc) - see $LOG/$label.log"
  finish 1 "$label failed"
}

commit() {   # commit <message> <files...>; only the named files; a failed push is logged and does not stop the run
  local msg=$1; shift
  git add -- "$@" || finish 1 "git add failed"
  if git diff --cached --quiet -- "$@"; then say "COMMIT nothing new: $msg"; return 0; fi
  git commit -q -m "$msg" -m "Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" -- "$@" || finish 1 "git commit failed"
  say "COMMIT $(git rev-parse --short HEAD) $msg"
  git push -q origin HEAD 2>> "$LOG/push.log" && say "PUSH  ok" || say "PUSH  failed (see $LOG/push.log); continuing"
}

say "RUN START seeds ${SEEDS:-1 2}, commit $(git rev-parse --short HEAD), dirty $(git status --porcelain -- . ':(exclude)results' | wc -l | tr -d ' ')"
for k in ${SEEDS:-1 2}; do
  R=results/seeds
  if [ "$k" != 1 ] && ! grep -q "DONE  train_s$k " "$S" && [ "$(date +%s)" -ge "$TRAIN2_BY" ]; then
    finish 4 "seed $k training not started: after the limit for a new training run"
  fi
  job "train_s$k" 1 tdd.stage2.train --stage 3 --group seeds --run "t2_20_s$k" --epochs 1 --small-aug 0.5 --cst-split cst_tr_20 --init runs/full_frcnn/best.pt --seed "$k"
  job "hold_s$k" 1 tdd.seeds.infer --seed "$k" --set cst_tr_hold
  job "vhold_s$k" 0 tdd.seeds.verify_caches --seed "$k" --scope hold
  job "freeze_s$k" 0 tdd.seeds.freeze --seed "$k"
  commit "feat(thermal-drone-detection): seed study, seed $k frozen (training record, hold-out cache check, threshold) before any test inference" \
    "$R/train_t2_20_s$k.json" "$R/cache_check__seed${k}_hold.json" "$R/frozen_seed$k.json"
  job "cstval_s$k" 1 tdd.seeds.infer --seed "$k" --set cst_val
  job "auv_s$k" 1 tdd.seeds.infer --seed "$k" --set auv_test
  job "auvs_s$k" 0 tdd.seeds.infer --seed "$k" --set auv_test_small
  job "svan_s$k" 1 tdd.seeds.infer --seed "$k" --set svan_test_h
  job "vtest_s$k" 0 tdd.seeds.verify_caches --seed "$k" --scope test
  job "eval_s$k" 0 tdd.seeds.evaluate --seed "$k"
  commit "feat(thermal-drone-detection): seed study, seed $k evaluated on CST val, Anti-UAV test and Svanström test" \
    "$R/cache_check__seed${k}_test.json" "$R/eval_seed$k.json"
done
finish 0 "all seeds done"
