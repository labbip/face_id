#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# FaceSim3D (2D / 3D conditions) Face-ID CV with 28_avfs_faceid.py: D0-D2 x 10 folds x seed 0. Logs logs/facesim_<tag>.log
cd "$ROOT"
PAR=${PAR:-4}; jobs=()
for c in 2D 3D; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do jobs+=("$c $k $f"); done; done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  uv run python -I scripts/28_avfs_faceid.py --path data/processed/facesim3d_$1.json --out results/facesim$1 --kind $2 --fold $3 --seed 0 --min_trip 100 --epochs 400 > logs/facesim$1_$2_f$3_s0.log 2>&1 &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait; echo DONE
