#!/bin/bash

# Script to train all model architectures
# Usage: bash run_training_all_models.sh

echo "Training all model architectures..."
echo "==================================="

# Train Simple CNN (for single frame classification)
echo ""
echo "1/3: Training Simple CNN..."
python train.py \
    --dataset_type frame_image \
    --model_type simple_cnn \
    --batch_size 32 \
    --max_epochs 30 \
    --experiment_name simple_cnn_experiment

# Train 3D CNN (for video classification)
echo ""
echo "2/3: Training 3D CNN..."
python train.py \
    --dataset_type frame_video \
    --model_type cnn3d \
    --batch_size 8 \
    --max_epochs 50 \
    --experiment_name cnn3d_experiment

# Train CNN-LSTM (for video classification with temporal modeling)
echo ""
echo "3/3: Training CNN-LSTM..."
python train.py \
    --dataset_type frame_video \
    --model_type cnn_lstm \
    --batch_size 8 \
    --max_epochs 50 \
    --hidden_size 256 \
    --num_layers 2 \
    --experiment_name cnn_lstm_experiment

echo ""
echo "All models trained!"
echo "Results saved in logs/ and checkpoints/ directories."

