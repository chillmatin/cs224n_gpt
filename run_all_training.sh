#!/bin/bash
# Run CFIMDB full-model training, then paraphrase detection sequentially.
# Run from the project root with conda env activated:
#   conda activate cs224n_dfp
#   bash run_all_training.sh

cd "$(dirname "$0")"

PYTHON=$(which python)

echo "=== Starting CFIMDB full-model training ==="
$PYTHON run_cfimdb_fullmodel.py 2>&1 | tee logs_cfimdb.txt
echo "=== CFIMDB done. Starting paraphrase detection ==="
$PYTHON run_paraphrase.py 2>&1 | tee logs_paraphrase.txt
echo "=== All training complete ==="
