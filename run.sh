#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES

source activate messou_env
cd /home/23r9802_chen/messou/TimeSerieForecasting/
python main.py
