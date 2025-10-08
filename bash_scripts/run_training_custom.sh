#!/bin/bash

# Script to run training with custom parameters
# Usage: bash run_training_custom.sh

echo "Starting Action Recognition Training with Custom Parameters..."
echo "=============================================================="

# Train with CNN3D model using Hydra overrides
python train.py \
    data.root_dir=data/ufc10 \
    data.dataset_type=frame_video \
    model.model_type=CNN3D \
    data.batch_size=8 \
    data.num_workers=4 \
    data.image_size=[112,112] \
    data.n_sampled_frames=16 \
    training.learning_rate=0.001 \
    training.weight_decay=0.0001 \
    training.max_epochs=50 \
    training.accelerator=auto \
    training.devices=1 \
    logging.experiment_name=cnn3d_experiment \
    training.patience=10

echo ""
echo "Training completed!"
echo "Check logs/ and checkpoints/ directories for results."

