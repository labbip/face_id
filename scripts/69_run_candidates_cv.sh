#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# CV runners for the 2026-10-28 candidates. usage: SRC=eva|jester|scut|scut_race [SEEDS="0 1 2"] [PAR=3] bash scripts/69_run_candidates_cv.sh
cd "$ROOT"
SRC=${SRC:-eva}; SEEDS=${SEEDS:-"0 1 2"}; PAR=${PAR:-3}; KINDS=${KINDS:-"D0 D1 D2"}; jobs=(); i=0
run() {  # kind fold seed
  case $SRC in
    eva)   uv run python -I scripts/53_lapis_faceid_img.py --kind $1 --fold $2 --seed $3 --min_rat 100 --out results/eva_img --data data/processed/eva_ratings.csv --ids results/eva_image_ids.npy --embed results/eva_embed_clip.npy ;;
    jester) uv run --with openpyxl --with xlrd python -I scripts/65_dense_faceid.py --src jester --kind $1 --fold $2 --seed $3 --epochs 300 --n_est 10,20,50 ;;
    scut)  uv run --with openpyxl --with xlrd python -I scripts/65_dense_faceid.py --src scut --kind $1 --fold $2 --seed $3 --epochs 600 --g_drop 0.2 --n_est 10,20,50,100,500 ;;
    scut_race) uv run --with openpyxl --with xlrd python -I scripts/65_dense_faceid.py --src scut --kind $1 --fold $2 --seed $3 --epochs 600 --g_drop 0.2 --n_est 10,20,50,100,500 --item_split $SPLIT --out results/scut_$SPLIT ;;
  esac
}
for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in $KINDS; do
  run $k $f $s > logs/${SRC}${SPLIT:+_$SPLIT}_$k\_f$f\_s$s.log 2>&1 &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done; done; done
wait; echo DONE
