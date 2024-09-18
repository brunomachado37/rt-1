#!/bin/bash
#SBATCH --job-name=eval_rt-1_robocasa
#SBATCH --output=logs/eval/eval-17-24_tasks-epoch-18.%j.out
#SBATCH --error=logs/eval/eval-17-24_tasks-epoch-18.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00

echo ${SLURM_NODELIST}

source ~/.bashrc
conda activate robocasa

python eval_robocasa.py --model_config conf/17_config_robocasa_tiny_h50.yaml \
                        --checkpoint_path runs/RT-1-RoboCasa/mjumurto/checkpoints/epoch=17-step=733014.ckpt \
                        --eval_config conf/eval/24_tasks \
						--output_dir eval \
                        --evaluations_per_task 30