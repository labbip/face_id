#!/bin/bash
cd "$(dirname "$0")/.."
run() { uv run python scripts/04_train.py --exp 1 --kind $1 --M $2 --fold 0 --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --direct --det --tag _det > logs/04_f0_$1_M$2_det.log 2>&1; }
run D1 4 &
run D1 16 &
run D2 4 &
run D3 4 &
run D3 16 &
wait
