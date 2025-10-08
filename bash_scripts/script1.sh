#!/bin/bash
#BSUB -j cv_job1
#BSUB -o hpc_outputs/cv_job1.out
#BSUB -e hpc_outputs/cv_job1.err
#BSUB -R "rusage[mem=1G]"
#BSUB -W 1:00
#BSUB -q gpuv100
#BSUB -B
#BSUB -N


source ~/miniconda3/bin/activate
conda activate computer_vision


python train.py
