#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# Effect-size metrics (residual r, contested-item r, contested-pair accuracy) for existing models (evaluation only) + London retrained in 65.
cd "$ROOT"; PAR=${PAR:-3}; i=0
wait_par() { i=$((i+1)); if (( i % PAR == 0 )); then wait; fi; }
for s in 0 1; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --eval_only > logs/eff_lapis_${k}_f${f}_s${s}.log 2>&1 & wait_par; done; done; done
for s in 0 1 2; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --eval_only --min_rat 100 --out results/eva_img --data data/processed/eva_ratings.csv --ids results/eva_image_ids.npy --embed results/eva_embed_clip.npy > logs/eff_eva_${k}_f${f}_s${s}.log 2>&1 & wait_par
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --eval_only --min_rat 100 --out results/aadb_img --data data/processed/aadb_ratings.csv --ids results/aadb_image_ids.npy --embed results/aadb_embed_clip.npy > logs/eff_aadb_${k}_f${f}_s${s}.log 2>&1 & wait_par
  uv run --with openpyxl --with xlrd python -I scripts/65_dense_faceid.py --src scut --kind $k --fold $f --seed $s --g_drop 0.2 --n_est 10,20,50,100,500 --eval_only > logs/eff_scut_${k}_f${f}_s${s}.log 2>&1 & wait_par; done; done; done
for s in 0 1; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do
  uv run --with openpyxl --with xlrd python -I scripts/65_dense_faceid.py --src jester --kind $k --fold $f --seed $s --n_est 10,20,50 --eval_only > logs/eff_jester_${k}_f${f}_s${s}.log 2>&1 & wait_par
  uv run python -I scripts/65_dense_faceid.py --src london --kind $k --fold $f --seed $s --epochs 300 --n_est 5,10,20,40 --out results/london65 > logs/eff_london_${k}_f${f}_s${s}.log 2>&1 & wait_par; done; done; done
wait; echo DONE
