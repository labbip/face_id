#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# Extra seeds (2-4) and D/M sensitivity (1 seed) for the generalization tests. PAR parallel. Logs logs/gsens_*.log
cd "$ROOT"
PAR=${PAR:-6}; jobs=()
# AVFS (fast): seeds 2-4 for D0-D3; sensitivity D=5/16, M=2/8 for D1 and D2 (seed 0)
for s in 2 3 4; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2 D3; do jobs+=("avfs $k $f $s 8 4"); done; done; done
for f in 0 1 2 3 4 5 6 7 8 9; do for k in D1 D2; do for dm in "5 4" "16 4" "8 2" "8 8"; do jobs+=("avfs $k $f 0 $dm"); done; done; done
# London learned embedding: seeds 2-4 for D0-D3; sensitivity for D1 and D2 (seed 0)
for s in 2 3 4; do for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2 D3; do jobs+=("london $k $f $s 8 4"); done; done; done
for f in 0 1 2 3 4 5 6 7 8 9; do for k in D1 D2; do for dm in "5 4" "16 4" "8 2" "8 8"; do jobs+=("london $k $f 0 $dm"); done; done; done
# London image embedding: seed 2 only for D0-D2
for f in 0 1 2 3 4 5 6 7 8 9; do for k in D0 D1 D2; do jobs+=("londonimg $k $f 2 8 4"); done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  case $1 in
    avfs) uv run python -I scripts/28_avfs_faceid.py --kind $2 --fold $3 --seed $4 --D $5 --M $6 --epochs 400 > logs/gsens_avfs_$2_D$5_M$6_f$3_s$4.log 2>&1 & ;;
    london) uv run python -I scripts/23_london_faceid.py --kind $2 --fold $3 --seed $4 --D $5 --M $6 --epochs 300 > logs/gsens_london_$2_D$5_M$6_f$3_s$4.log 2>&1 & ;;
    londonimg) uv run python -I scripts/31_london_faceid_img.py --kind $2 --fold $3 --seed $4 --D $5 --M $6 --epochs 300 --embed clip --pcs 32 > logs/gsens_londonimg_$2_D$5_M$6_f$3_s$4.log 2>&1 & ;;
  esac
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait; echo DONE
