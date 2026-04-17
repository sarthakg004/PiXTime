#!/bin/bash

set -e  # Exit immediately if any command fails

echo "🔧 Making all scripts executable..."

chmod +x dlinear_run.sh
chmod +x itrans_run.sh
chmod +x patchtst_run.sh
chmod +x pixtime_run.sh
chmod +x timexer_run.sh

echo "🚀 Starting sequential execution..."

echo "▶ Running DLinear..."
./dlinear_run.sh
echo "✅ DLinear completed"

echo "▶ Running iTransformer..."
./itrans_run.sh
echo "✅ iTransformer completed"

echo "▶ Running PatchTST..."
./patchtst_run.sh
echo "✅ PatchTST completed"

echo "▶ Running PiXTime..."
./pixtime_run.sh
echo "✅ PiXTime completed"

echo "▶ Running TimeXer..."
./timexer_run.sh
echo "✅ TimeXer completed"

echo "🎉 All scripts executed successfully!"