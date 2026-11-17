#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# XPASS follow-ups (2026-10-29): (A) transfer: LAPIS models -> XPASS raters (results/lapis_on_xpass), XPASS art models -> LAPIS raters on the 2,345 images (results/xpass_on_lapis);
# (B) LAPIS size controls: 129 random raters, all ratings (results/lapis_r129_img, --max_raters) and 129 raters x 227 ratings (results/lapis_thin129_img);
# (C) rating-column variants Like / Beautiful for the three XPASS domains (74 --col, 75 COL=).  usage: PAR=3 bash scripts/77_run_xpass_followups.sh
cd "$ROOT"; PAR=${PAR:-3}; i=0
wp() { i=$((i+1)); if (( i % PAR == 0 )); then wait; fi; }
for s in 0 1; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --eval_only --eval_write --out results/lapis_on_xpass --data data/processed/xpass_art.csv > logs/lapis_on_xpass_${k}_f${f}_s${s}.log 2>&1 & wp
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --eval_only --eval_write --min_rat 60 --out results/xpass_on_lapis --data data/processed/lapis_on_xpassimgs.csv --ids results/xpass_art_image_ids.npy --embed results/xpass_art_embed_clip.npy > logs/xpass_on_lapis_${k}_f${f}_s${s}.log 2>&1 & wp
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --max_raters 129 --out results/lapis_r129_img > logs/lapis_r129_${k}_f${f}_s${s}.log 2>&1 & wp
  uv run python -I scripts/53_lapis_faceid_img.py --kind $k --fold $f --seed $s --out results/lapis_thin129_img --data data/processed/lapis_thin129_s${s}.csv > logs/lapis_thin129_${k}_f${f}_s${s}.log 2>&1 & wp
done; done; done; wait
uv run python -I scripts/74_xpass_prep.py --col Like > logs/74_xpass_prep_Like.log 2>&1; uv run python -I scripts/74_xpass_prep.py --col Beautiful > logs/74_xpass_prep_Beautiful.log 2>&1
COL=Like PAR=$PAR bash scripts/75_run_xpass_cv.sh > logs/75_run_xpass_Like.log 2>&1; COL=Beautiful PAR=$PAR bash scripts/75_run_xpass_cv.sh > logs/75_run_xpass_Beautiful.log 2>&1
for t in lapis_on_xpass xpass_on_lapis lapis_r129_img lapis_thin129_img xpass_art_Like_img xpass_fashion_Like_img xpass_scenery_Like_img xpass_art_Beautiful_img xpass_fashion_Beautiful_img xpass_scenery_Beautiful_img; do uv run python -I scripts/33_london_img_analyze.py $t > logs/33_$t.log 2>&1; done
echo DONE
