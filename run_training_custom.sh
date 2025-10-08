#!/bin/bash

# Script to run training with custom parameters
# Usage: bash run_training_custom.sh

echo "Starting Action Recognition Training with Custom Parameters..."
echo "=============================================================="

# Train with CNN3D model
python train.py \
    --root_dir data/ufc10 \
    --dataset_type frame_video \
    --model_type cnn3d \
    --batch_size 8 \
    --num_workers 4 \
    --image_size 112 112 \
    --n_sampled_frames 16 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --max_epochs 50 \
    --accelerator auto \
    --devices 1 \
    --experiment_name cnn3d_experiment \
    --patience 10

echo ""
echo "Training completed!"
echo "Check logs/ and checkpoints/ directories for results."

