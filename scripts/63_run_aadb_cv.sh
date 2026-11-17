#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# AADB photo aesthetics Face-ID (image version, scripts/53 with AADB paths): D0-D2 x 10 rater folds x seeds. Logs logs/aadb_img_<tag>.log
cd "$ROOT"
SEEDS=${SEEDS:-"0 1 2"}; PAR=${PAR:-4}; MINRAT=${MINRAT:-100}; OUT=${OUT:-results/aadb_img}; KINDS=${KINDS:-"D0 D1 D2"}; jobs=()
for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in $KINDS; do jobs+=("$k $f $s"); done; done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  uv run python -I scripts/53_lapis_faceid_img.py --kind $1 --fold $2 --seed $3 --min_rat $MINRAT --out $OUT --data data/processed/aadb_ratings.csv --ids results/aadb_image_ids.npy --embed results/aadb_embed_clip.npy > logs/aadb_img_$1_f$2_s$3.log 2>&1 &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait; echo DONE
