#!/bin/bash

# Script to train all model architectures
# Usage: bash run_training_all_models.sh

echo "Training all model architectures..."
echo "==================================="

# Train Single Frame Model (for single frame classification)
echo ""
echo "1/5: Training Single Frame Model..."
python train.py --config-name=experiment_single_frame

# Train Early Fusion Model (for single frame classification)
echo ""
echo "2/5: Training Early Fusion Model..."
python train.py --config-name=experiment_early_fusion

# Train Late Fusion Model (for single frame classification)
echo ""
echo "3/5: Training Late Fusion Model..."
python train.py --config-name=experiment_late_fusion

# Train 3D CNN (for video classification)
echo ""
echo "4/5: Training 3D CNN..."
python train.py --config-name=experiment_cnn3d

# Train C3D (for video classification)
echo ""
echo "5/5: Training C3D..."
python train.py --config-name=experiment_c3d

echo ""
echo "All models trained!"
echo "Results saved in logs/ and checkpoints/ directories."

