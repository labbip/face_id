#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# London Set image-embedding Face-ID CV (CLIP, 32 PCs): D0-D3 x 10 folds x seeds, 4 parallel. Logs logs/london_img_<tag>.log
cd "$ROOT"
SEEDS=${SEEDS:-"0 1"}
jobs=()
for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2 D3; do jobs+=("$k $f $s"); done; done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  uv run python -I scripts/31_london_faceid_img.py --kind $1 --fold $2 --seed $3 --epochs 300 --embed clip --pcs 32 > logs/london_img_$1_f$2_s$3.log 2>&1 &
  i=$((i+1)); if (( i % ${PAR:-4} == 0 )); then wait; fi
done
wait; echo DONE
