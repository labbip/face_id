#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# D1-type vs D2-type simulated participants x D1/D2 models, exp1 and exp2 fold-0 D0 embeddings, 2 seeds. 4 parallel.
cd "$ROOT"
i=0
for e in 1 2; do for s in 0 1; do for sim in D1 D2; do for k in D1 D2; do
  uv run python -I scripts/48_sim_recovery_d2.py results/runs/exp${e}_D0_D8_M4_f0_s0_cv --sim $sim --kind $k --M 8 --seed $s > logs/sim_recovery_exp${e}_sim${sim}_kind${k}_s${s}.log 2>&1 &
  i=$((i+1)); if (( i % 4 == 0 )); then wait; fi
done; done; done; done
wait; echo DONE
