#!/bin/bash

set -euo pipefail

# Run all PiXTime baseline + enhancement configurations and save detailed summaries.
# Optional environment variables:
#   PYTHON_BIN   (default: ./.venv/bin/python if available, else python)
#   TRAIN_EPOCHS (default: 10)

if [[ -x "./.venv/bin/python" ]]; then
	PYTHON_BIN=${PYTHON_BIN:-./.venv/bin/python}
else
	PYTHON_BIN=${PYTHON_BIN:-python}
fi
TRAIN_EPOCHS=${TRAIN_EPOCHS:-10}

echo "Preparing executable scripts..."
chmod +x pixtime_enhanced_run.sh

mkdir -p results
RUN_TS=$(date +"%Y%m%d_%H%M%S")

echo "Starting PiXTime baseline + enhancement grid..."
PYTHON_BIN="$PYTHON_BIN" TRAIN_EPOCHS="$TRAIN_EPOCHS" ./pixtime_enhanced_run.sh
echo "Grid execution completed."

echo "Generating detailed summary report..."
"$PYTHON_BIN" evaluate_improvements.py | tee "results/summary_${RUN_TS}.txt"
cp "results/summary_${RUN_TS}.txt" "results/summary.txt"

echo "Detailed summary saved to: results/summary.txt"
echo "Timestamped summary saved to: results/summary_${RUN_TS}.txt"
echo "All run metrics available at: results/enhanced_results.csv"

echo "All baseline + enhancement experiments completed successfully."