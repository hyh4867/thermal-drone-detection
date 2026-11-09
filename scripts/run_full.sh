#!/usr/bin/env bash
# Full run (4 epochs, full evaluation). Run from the project root:
#   caffeinate -i bash scripts/run_full.sh
# Each stage logs to results/full/logs/<stage>.log; on failure the script stops,
# keeping every result and prediction cache written so far.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=.venv/bin/python
CFG=configs/full.yaml
LOG=results/full/logs
mkdir -p "$LOG"
stage() {
  local name=$1; shift
  if grep -q "DONE  $name" "$LOG/stages.log" 2>/dev/null; then
    echo "$(date '+%F %T %Z') SKIP  $name (already DONE)" | tee -a "$LOG/stages.log"; return 0
  fi
  echo "$(date '+%F %T %Z') START $name" | tee -a "$LOG/stages.log"
  local t0=$SECONDS
  if "$@" >> "$LOG/$name.log" 2>&1; then
    echo "$(date '+%F %T %Z') DONE  $name ($(( (SECONDS - t0) / 60 )) min)" | tee -a "$LOG/stages.log"
  else
    echo "$(date '+%F %T %Z') FAIL  $name ($(( (SECONDS - t0) / 60 )) min) — see $LOG/$name.log" | tee -a "$LOG/stages.log"
    exit 1
  fi
}
echo "$(date '+%F %T %Z') RUN full, commit $(git rev-parse --short HEAD), memory $(sysctl -n hw.memsize) B" | tee -a "$LOG/stages.log"
stage 0_tests_$(date +%H%M) $PY -m pytest -q tests
stage 1_baseline     $PY -m tdd.baseline   --config $CFG --out results/full/baseline --run-name auto
stage 2_train        $PY -m tdd.train      --config $CFG --run full_frcnn --results-dir results/full/model --resume
stage 3_eval_model   $PY -m tdd.eval_model --config $CFG --run full_frcnn --out results/full/model
stage 4_report       $PY -m tdd.report_full
echo "$(date '+%F %T %Z') ALL DONE" | tee -a "$LOG/stages.log"
