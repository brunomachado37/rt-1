#!/bin/bash
#SBATCH --job-name=rt-1_robocasa_train
#SBATCH --output=logs/train/train.%j.out
#SBATCH --error=logs/train/train.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=90G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00

echo ${SLURM_NODELIST}

source ~/.bashrc
conda activate robocasa

python train_robocasa.py --config-name 12_config_robocasa_lr-1e-3