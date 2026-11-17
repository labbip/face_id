#!/bin/bash
cd "$(dirname "$0")/.."
for exp in 1 2; do for f in 0 1 2 3 4 5 6 7 8 9; do echo "uv run python scripts/05_baselines.py --exp $exp --fold $f > logs/05_baselines_exp${exp}_f${f}.log 2>&1"; done; done | xargs -P 3 -I{} bash -c "{}"
echo DONE
