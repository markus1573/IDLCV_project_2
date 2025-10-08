#!/bin/bash

# Script to train all model architectures
# Usage: bash run_training_all_models.sh

echo "Training all model architectures..."
echo "==================================="

# Train Single Frame Model (for single frame classification)
echo ""
echo "1/4: Training Single Frame Model..."
python train.py --config-name=experiment_single_frame

# Train Early Fusion Model (for single frame classification)
echo ""
echo "2/4: Training Early Fusion Model..."
python train.py --config-name=experiment_early_fusion

# Train Late Fusion Model (for single frame classification)
echo ""
echo "3/4: Training Late Fusion Model..."
python train.py --config-name=experiment_late_fusion

# Train 3D CNN (for video classification)
echo ""
echo "4/4: Training 3D CNN..."
python train.py --config-name=experiment_cnn3d

echo ""
echo "All models trained!"
echo "Results saved in logs/ and checkpoints/ directories."

