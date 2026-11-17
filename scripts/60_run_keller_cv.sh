#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# Keller & Vosshall odor pleasantness Face-ID CV: D0-D3 x 10 subject folds x seeds. Logs logs/keller_<tag>.log
# usage: SEEDS="0 1 2" PAR=4 CONC=high EMBED=desc nohup bash scripts/60_run_keller_cv.sh > logs/60_keller_cv.log 2>&1 &
cd "$ROOT"
SEEDS=${SEEDS:-"0 1 2"}; PAR=${PAR:-4}; CONC=${CONC:-high}; EMBED=${EMBED:-desc}; OUT=${OUT:-results/keller}; jobs=()
for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2 D3; do jobs+=("$k $f $s"); done; done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  uv run --with openpyxl python -I scripts/59_keller_faceid.py --kind $1 --fold $2 --seed $3 --conc $CONC --embed $EMBED --out $OUT > logs/keller_${CONC}_${EMBED}_$1_f$2_s$3.log 2>&1 &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait; echo DONE
