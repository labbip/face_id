#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# 2026-11-17: simulated-participant control at the measured reliability. 6 processes in parallel (GPU use is small).
#   matched    = noise calibrated to the real est-vs-hold rho (Exp. 1 and 2), wsd 0.6 as before
#   matched_id = noise AND size of individual differences calibrated (real est-vs-hold rho and real held-out rho of the population RDM), Exp. 2
# Run: nohup bash scripts/88_run_sim_recovery_matched.sh > logs/88_sim_recovery_matched.log 2>&1 &
cd "$ROOT"
for s in 0 1; do
  for e in 1 2; do uv run python -I scripts/88_sim_recovery_matched.py --exp $e --seed $s --tag matched > logs/88_matched_exp${e}_s${s}.log 2>&1 & done
  uv run python -I scripts/88_sim_recovery_matched.py --exp 2 --seed $s --tag matched_id --match_pop > logs/88_matched_id_exp2_s${s}.log 2>&1 &
done
wait; echo DONE
