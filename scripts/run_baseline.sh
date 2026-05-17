#!/usr/bin/env bash
set -euo pipefail

python main.py --dataset dataset1 --epochs 70 --early_stop 12 --batch_size 200 --num_neighbors 30
python main.py --dataset dataset2 --epochs 20 --early_stop 5 --batch_size 200 --num_neighbors 30
