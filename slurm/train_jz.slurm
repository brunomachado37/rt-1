#!/bin/bash

#SBATCH --job-name=rt1_pushcube
#SBATCH --output=logs/rt1_pushcube.%j.out
#SBATCH --error=logs/rt1_pushcube.%j.err
#SBATCH -A uli@v100
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}
source ~/.bashrc

module purge
module load pytorch-gpu/py3/2.0.0

export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1 

srun python train_oc.py
