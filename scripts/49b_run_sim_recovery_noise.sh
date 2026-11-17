#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# Power vs reliability: same simulation with lower noise (split-half ~0.35, like Xu Exp 2) on exp2 fold-0 embedding. 4 parallel.
cd "$ROOT"
mkdir -p results/sim_recovery_n03
i=0
for s in 0 1; do for sim in D1 D2; do for k in D1 D2; do
  uv run python -I scripts/48_sim_recovery_d2.py results/runs/exp2_D0_D8_M4_f0_s0_cv --sim $sim --kind $k --M 8 --seed $s --noise 0.3 > logs/sim_recovery_n03_exp2_sim${sim}_kind${k}_s${s}.log 2>&1 &
  i=$((i+1)); if (( i % 4 == 0 )); then wait; fi
done; done; done
wait; echo DONE
