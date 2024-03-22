#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
pwd

# Activate conda environment
source activate time_serie_torch

# Determine the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change directory to the script directory
cd "$SCRIPT_DIR"

# Pass the model_group parameter to the Python script
python src/main.py --model_group 1
python src/main.py --model_group 2
