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
import argparse
import logging
import os
import platform
import random
import socket

import numpy as np
import psutil
import torch

from models.model_validation import ComprehensiveModelValidator
from utils.constants import (
    ELECTRICITY, LOOK_BACKS, FORECAST_PERIODS, SEEDER, LOG_FILE,
    LSTM_MODEL, GRU_MODEL, CNN_MODEL, CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_MODEL, CNN_ATTENTION_GRU_MODEL,
    CNN_ATTENTION_BiLSTM_MODEL, CNN_ATTENTION_BiGRU_MODEL, APARTMENT, HOUSE, ENERGY, SIMPLE_OR_AUGMENTED,
    UNI_OR_MULTI_VARIATE, CNN_BiGRU_ATTENTION_MODEL, CNN_BiLSTM_ATTENTION_MODEL, )

# Set seed for reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.manual_seed(SEEDER)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(SEEDER)

np.random.seed(SEEDER)

'''----------------------------------------------------------------------------------------------'''


def main(model_group, dataset):
    if dataset == 'electricity':
        series = ELECTRICITY
    elif dataset == 'house':
        series = HOUSE
    elif dataset == 'apartment':
        series = APARTMENT
    elif dataset == 'energy':
        series = ENERGY
    else:
        series = ELECTRICITY

    model_group1 = [CNN_MODEL, LSTM_MODEL, GRU_MODEL]

    model_group2 = [CNN_LSTM_MODEL, CNN_GRU_MODEL]

    model_group4 = [
                    CNN_LSTM_ATTENTION_MODEL,
                    CNN_GRU_ATTENTION_MODEL,
                    CNN_BiLSTM_ATTENTION_MODEL,
                    CNN_BiGRU_ATTENTION_MODEL,
                    ]
    model_group5 = [
                    CNN_ATTENTION_LSTM_MODEL,
                    CNN_ATTENTION_GRU_MODEL,
                    CNN_ATTENTION_BiLSTM_MODEL,
                    CNN_ATTENTION_BiGRU_MODEL
                    ]
    model_group0 = [CNN_ATTENTION_GRU_MODEL]

    look_backs = LOOK_BACKS  # [7]
    forecast_periods = FORECAST_PERIODS  # [3]

    # Determine model types based on model_group
    if model_group == 1:
        model_types = model_group1
    elif model_group == 2:
        model_types = model_group2
    elif model_group == 4:
        model_types = model_group4
    elif model_group == 5:
        model_types = model_group5
    else:
        model_types = model_group0

    logger.info(f"model_group: {model_group} |||| model_types: {model_types}")

    # Create ModelTuner instance and Optuna study
    # if model_group != 0:
    # model_tuner_and_study(look_backs, forecast_periods, model_types, series)

    # Build best model
    # trainer = ComprehensiveModelTrainer(look_backs=look_backs, forecast_periods=forecast_periods,
    #                                     model_types=model_types, series_types=series)
    # trainer.build_and_train_models()

    trainer = ComprehensiveModelValidator(look_backs=look_backs, forecast_periods=forecast_periods,
                                          model_types=model_types, series_types=series)
    trainer.build_and_train_models()


'''----------------------------------------------------------------------------------------------'''

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
        logger.info(f"CPU Name: {cpu_model}")
        logger.info(f"CPU Cores: {cpu_count}")


'''----------------------------------------------------------------------------------------------'''

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_group", type=int, help="Model group parameter: group of models to process")
    parser.add_argument("--dataset", type=str, default="default_dataset", help="Dataset parameter: dataset to use")
    args = parser.parse_args()
    log_filename = (LOG_FILE + SIMPLE_OR_AUGMENTED +
                    '/' + str(UNI_OR_MULTI_VARIATE) +  # Convert UNI_OR_MULTI_VARIATE to a string
                    '_' + str(args.dataset or ELECTRICITY) +
                    '_group' + str(args.model_group or 0) +
                    '.log')

    # Logging settings config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )

    # Call the function to trigger logging
    some_function()
    current_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"CURRENT PATH IS: {current_dir}")

    # Call the main function and pass the model_group parameter
    main(args.model_group or 0, args.dataset or ELECTRICITY)
