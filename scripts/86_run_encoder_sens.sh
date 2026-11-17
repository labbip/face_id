#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# Encoder sensitivity (2026-11-12): same CV as the CLIP ViT-B/32 image versions (same folds, seeds, D = 8, M = 4, PCA 64; London PCA 32)
# with the embeddings of scripts/84 (ENCS: dinov2b, clipl).  Outputs results/<ds>_img_<enc>/, logs logs/enc_<ds>_<enc>_<tag>.log.
# usage: [ENCS="dinov2b clipl"] [DSS="lapis eva aadb london"] [PAR=4] bash scripts/86_run_encoder_sens.sh   (from the project root)
cd "$ROOT"
ENCS=${ENCS:-"dinov2b clipl"}; DSS=${DSS:-"lapis eva aadb london"}; PAR=${PAR:-4}; i=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}   # PCA of 768-dim embeddings otherwise oversubscribes the CPUs
done_() { case $1 in london) [ -f results/london_img_$2/model_$3_D8_M4_$2_f$4_s$5.pt ] ;; *) [ -f results/$1_img_$2/model_$3_D8_M4_f$4_s$5.pt ] ;; esac; }   # skip finished runs (restartable)
run() {  # ds enc kind fold seed
  case $1 in
    lapis)  uv run python -I scripts/53_lapis_faceid_img.py --kind $3 --fold $4 --seed $5 --epochs 300 --out results/lapis_img_$2 --embed results/lapis_embed_$2.npy ;;
    eva)    uv run python -I scripts/53_lapis_faceid_img.py --kind $3 --fold $4 --seed $5 --min_rat 100 --out results/eva_img_$2 --data data/processed/eva_ratings.csv --ids results/eva_image_ids.npy --embed results/eva_embed_$2.npy ;;
    aadb)   uv run python -I scripts/53_lapis_faceid_img.py --kind $3 --fold $4 --seed $5 --min_rat 100 --out results/aadb_img_$2 --data data/processed/aadb_ratings.csv --ids results/aadb_image_ids.npy --embed results/aadb_embed_$2.npy ;;
    london) uv run python -I scripts/31_london_faceid_img.py --kind $3 --fold $4 --seed $5 --epochs 300 --embed $2 --pcs 32 --out results/london_img_$2 ;;
  esac
}
for ds in $DSS; do
  case $ds in lapis) SEEDS="0 1" ;; *) SEEDS="0 1 2" ;; esac     # same seeds as the CLIP ViT-B/32 runs
  for enc in $ENCS; do for s in $SEEDS; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do
    done_ $ds $enc $k $f $s && continue
    run $ds $enc $k $f $s > logs/enc_${ds}_${enc}_${k}_f${f}_s${s}.log 2>&1 &
    i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
  done; done; done; done
done
wait; echo DONE
