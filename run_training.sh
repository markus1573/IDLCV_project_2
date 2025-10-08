#!/bin/bash

# Script to run training with default settings
# Usage: bash run_training.sh

echo "Starting Action Recognition Training..."
echo "========================================"

# Train with config file
python train_from_config.py \
    --config config.yaml

echo ""
echo "Training completed!"
echo "Check logs/ and checkpoints/ directories for results."

