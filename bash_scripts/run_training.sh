#!/bin/bash

# Script to run training with default settings
# Usage: bash run_training.sh

echo "Starting Action Recognition Training..."
echo "========================================"

# Train with default config
python train.py

echo ""
echo "Training completed!"
echo "Check experiments/<model>/<date>/<time>/ for logs, checkpoints and outputs."

