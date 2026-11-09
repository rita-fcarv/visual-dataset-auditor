#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results/processed
mkdir -p results/optimized_validation
mkdir -p results/benchmarks

make

python3 scripts/generate_dataset.py
python3 scripts/generate_validation.py
python3 scripts/generate_robustness_dataset.py

./visual_auditor   --input data/images   --output results/processed   --threshold 5

./visual_auditor   --input data/robustness   --output results/optimized_validation   --threshold 5

python3 scripts/evaluate_similarity.py   --input data/robustness   --report results/optimized_validation/similarity_report.csv

echo "Dataset auditing and validation completed."
