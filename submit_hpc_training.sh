#!/bin/bash

#BSUB -q gpuv100
#BSUB -J action_recognition[1-4]
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o hpc_training_%J_%I.out
#BSUB -e hpc_training_%J_%I.err
#BSUB -B
#BSUB -N


module load python3/3.12.11 
# Initialize conda environment
source idlcv_venv/bin/activate

cd IDLCV_project_2

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
    5)
        echo "Job $LSB_JOBINDEX: Training FlowResNet Model"
        python train.py --config-name=experiment_flow_resnet
        ;;

    *)
        echo "Invalid job index: $LSB_JOBINDEX"
        exit 1
        ;;
esac

echo "Job $LSB_JOBINDEX completed successfully"
