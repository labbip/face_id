#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# T3 runner: Face-ID D0-D2 x 10 folds x seeds on the LAPIS-like synthetic raters (81) with LAPIS ids/embeddings; then 33 summary and 55 sign check.
# usage: CONDS="cont disc" SEEDS="0 1" PAR=4 bash scripts/82_run_sim_lapis.sh
cd "$ROOT"; CONDS=${CONDS:-"cont disc"}; SEEDS=${SEEDS:-"0 1"}; PAR=${PAR:-4}; i=0
for c in $CONDS; do for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --out results/sim_lapis_$c --data data/processed/sim_lapis_$c.csv > logs/sim_lapis_${c}_$k\_f$f\_s$s.log 2>&1 &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done; done; done; done; wait
for c in $CONDS; do uv run python -I scripts/33_london_img_analyze.py sim_lapis_$c > logs/33_sim_lapis_$c.log 2>&1; uv run python -I scripts/55_sign_check.py sim_lapis_$c > logs/55_sim_lapis_$c.log 2>&1; done; echo DONE
