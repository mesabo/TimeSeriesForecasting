# main.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/03/2024
🚀 Welcome to the Awesome Python Script 🚀

User: mesabo
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab

"""
import logging
import os
import platform
import random
import socket

import numpy as np
import psutil
import torch

from hyperparameter_tuning.model_tuner_study import model_tuner_and_study
from input_processing.data_processing import preprocess_augment_and_split_dataset
from models.model_training import (build_best_model)
from utils.constants import (
    CNN_LSTM_ATTENTION_MODEL, ELECTRICITY, LOOK_BACKS, FORECAST_PERIODS, SEEDER
)

# Set seed for reproducibility
torch.manual_seed(SEEDER)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(SEEDER)

np.random.seed(SEEDER)

'''----------------------------------------------------------------------------------------------'''


def main():
    series = [ELECTRICITY]
    model_types = [CNN_LSTM_ATTENTION_MODEL]

    look_backs = LOOK_BACKS  # [7]
    forecast_periods = FORECAST_PERIODS  # [3]

    # Create ModelTuner instance and Optuna study
    model_tuner, study, best_params = model_tuner_and_study(look_backs, forecast_periods, model_types, series)

    # Build best model
    X_train, X_val, y_train, y_val, scaler = preprocess_augment_and_split_dataset(ELECTRICITY, 'D', look_backs[0],
                                                                                  forecast_periods[0])
    input_dim = X_train.shape[2]
    build_best_model(study, X_train, y_train, X_val, y_val, scaler, input_dim,
                     look_backs[0], forecast_periods[0], model_types[0],
                     series[0])


'''----------------------------------------------------------------------------------------------'''

# Logging settings config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='time_serie_gpu.log',
    filemode='w'
)

logger = logging.getLogger(__name__)

# Add hostname to log messages
hostname = socket.gethostname()

# Get GPU information if available
gpu_info = None
cpu_info, cpu_count = None, None
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_properties(torch.cuda.current_device())
else:
    cpu_count = psutil.cpu_count(logical=True)
    cpu_model = platform.processor()


# Example usage of the logger
def some_function():
    logger.info(f"Running on host: {hostname}")
    if torch.cuda.is_available():
        logger.info("Tuning on GPU server")
        logger.info(f"GPU Device Name: {gpu_info.name}")
        logger.info(f"GPU Memory Total: {gpu_info.total_memory} bytes")
    else:
        logger.info("Tuning on CPU server")
        logger.info(f"CPU Name: {cpu_count}")
        logger.info(f"CPU Cores: {cpu_model}")


'''----------------------------------------------------------------------------------------------'''

if __name__ == "__main__":
    # Call the function to trigger logging
    some_function()
    current_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"CURRENT PATH IS: {current_dir}")
    main()
