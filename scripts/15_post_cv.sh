#!/bin/bash
cd "$(dirname "$0")/.."
(uv run python scripts/14_learning_curve.py --exp 1 --kind D2 --M 4 --tag _cv > logs/14_lc_exp1_D2M4.log 2>&1; uv run python scripts/14_learning_curve.py --exp 1 --kind D3 --M 16 --tag _cv > logs/14_lc_exp1_D3M16.log 2>&1) &
(uv run python scripts/14_learning_curve.py --exp 2 --kind D1 --M 4 --tag _cv > logs/14_lc_exp2_D1M4.log 2>&1; uv run python scripts/14_learning_curve.py --exp 2 --kind D2 --M 4 --tag _cv > logs/14_lc_exp2_D2M4.log 2>&1) &
(uv run python scripts/11_z_attributes.py --exp 2 --kind D1 --M 4 --tag _cv --infer amortized > logs/11_h5_exp2_D1M4.log 2>&1; uv run python scripts/11_z_attributes.py --exp 2 --kind D2 --M 4 --tag _cv --infer amortized > logs/11_h5_exp2_D2M4.log 2>&1; uv run python scripts/11_z_attributes.py --exp 1 --kind D2 --M 4 --tag _cv --infer amortized > logs/11_h5_exp1_D2M4.log 2>&1) &
(for f in 0 1 2 3 4 5 6 7 8 9; do uv run --with sentence-transformers python scripts/12_text_to_z.py --exp 1 --kind D2 --M 4 --tag _cv --fold $f 2>&1 | grep -v "Warn\|Loading"; done > logs/12_h6_exp1_D2M4.log 2>&1; for f in 0 1 2 3 4 5 6 7 8 9; do uv run --with sentence-transformers python scripts/12_text_to_z.py --exp 2 --kind D1 --M 4 --tag _cv --fold $f 2>&1 | grep -v "Warn\|Loading"; done > logs/12_h6_exp2_D1M4.log 2>&1) &
(uv run python scripts/10_cross_individual.py results/runs/exp1_D2_D8_M4_f0_s0_cv > logs/10_cross_exp1_D2M4_cv.log 2>&1; uv run python scripts/10_cross_individual.py results/runs/exp2_D1_D8_M4_f0_s0_cv > logs/10_cross_exp2_D1M4_cv.log 2>&1) &
wait; echo DONE
