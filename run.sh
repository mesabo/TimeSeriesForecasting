#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
pwd
source activate time_serie_torch
cd /home/23r9802_chen/messou/TimeSerieForecasting/src/
python main.py
