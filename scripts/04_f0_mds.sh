#!/bin/bash
# fold-0 tests of the improved configuration (MDS init, residual encoder features, direct z inference)
cd "$(dirname "$0")/.."
run() { uv run python scripts/04_train.py --exp 1 --kind $1 --M $2 --fold 0 --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --direct --tag _mds > logs/04_f0_$1_M$2_mds.log 2>&1; }
run D0 4 &
run D1 4 &
run D1 16 &
run D3 4 &
run D2 4 &
wait
