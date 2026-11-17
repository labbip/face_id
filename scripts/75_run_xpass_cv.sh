#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# XPASS-Vis Face-ID CV (scripts/53 with XPASS paths): D0-D2 x 10 rater folds x seeds, per genre. Logs logs/xpass_<genre>_<tag>.log
# usage: GENRES="art fashion scenery" SEEDS="0 1" PAR=3 bash scripts/75_run_xpass_cv.sh
cd "$ROOT"
GENRES=${GENRES:-"art fashion scenery"}; SEEDS=${SEEDS:-"0 1"}; PAR=${PAR:-3}; KINDS=${KINDS:-"D0 D1 D2"}; COL=${COL:-}; SFX=${COL:+_$COL}; i=0   # COL=Like -> rating column variant from 74 --col Like
for g in $GENRES; do for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in $KINDS; do
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --min_rat 100 --out results/xpass_${g}${SFX}_img --data data/processed/xpass_$g$SFX.csv --ids results/xpass_${g}_image_ids.npy --embed results/xpass_${g}_embed_clip.npy > logs/xpass_${g}${SFX}_$k\_f$f\_s$s.log 2>&1 &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done; done; done; done
wait; echo DONE
