#!/bin/bash

# LSF Job Array Script for Parallel Model Training
# This script submits all model types as parallel jobs on HPC

# Job array parameters - one job per model type
#BSUB -J action_recognition[1-4]
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -o logs/hpc_training_%J_%I.out
#BSUB -e logs/hpc_training_%J_%I.err

# Initialize conda environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Create logs directory if it doesn't exist
mkdir -p logs

# Define model configurations based on job array index
case $LSB_JOBINDEX in
    1)
        echo "Job $LSB_JOBINDEX: Training Single Frame Model"
        python train.py --config-name=experiment_single_frame
        ;;
    2)
        echo "Job $LSB_JOBINDEX: Training Early Fusion Model"
        python train.py --config-name=experiment_early_fusion
        ;;
    3)
        echo "Job $LSB_JOBINDEX: Training Late Fusion Model"
        python train.py --config-name=experiment_late_fusion
        ;;
    4)
        echo "Job $LSB_JOBINDEX: Training CNN3D Model"
        python train.py --config-name=experiment_cnn3d
        ;;

    *)
        echo "Invalid job index: $LSB_JOBINDEX"
        exit 1
        ;;
esac

echo "Job $LSB_JOBINDEX completed successfully"
