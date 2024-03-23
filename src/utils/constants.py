#!/usr/bin/env python3
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

# Simple models
LSTM_MODEL = "LSTM-based"
GRU_MODEL = "GRU-based"
CNN_MODEL = "CNN-Based"

# Bi models
BiLSTM_MODEL = "BiLSTM-based"
BiGRU_MODEL = "BiGRU-based"

# Simple models + Attention
LSTM_ATTENTION_MODEL = "LSTM-Attention-based"
GRU_ATTENTION_MODEL = "GRU-Attention-based"
CNN_ATTENTION_MODEL = "CNN-Attention-Based"

# Bi models + Attention
BiLSTM_ATTENTION_MODEL = "BiLSTM-Attention-based"
BiGRU_ATTENTION_MODEL = "BiGRU-Attention-based"

# Hybrid models
CNN_LSTM_MODEL = "CNN-LSTM-based"
CNN_GRU_MODEL = "CNN-GRU-based"
CNN_BiLSTM_MODEL = "CNN-BiLSTM-based"
CNN_BiGRU_MODEL = "CNN-BiGRU-based"
CNN_LSTM_ATTENTION_MODEL = "CNN-LSTM-Attention-based"
CNN_GRU_ATTENTION_MODEL = "CNN-GRU-Attention-based"
CNN_BiLSTM_ATTENTION_MODEL = "CNN-BiLSTM-Attention-based"
CNN_BiGRU_ATTENTION_MODEL = "CNN-BiGRU-Attention-based"

# Custom Hybrid models
CNN_LSTM_ATTENTION_LSTM_MODEL = "CNN-LSTM-Attention-LSTM-based"
CNN_GRU_ATTENTION_GRU_MODEL = "CNN-GRU-Attention-GRU-based"
CNN_BiLSTM_ATTENTION_BiLSTM_MODEL = "CNN-BiLSTM-Attention-BiLSTM-based"
CNN_BiGRU_ATTENTION_BiGRU_MODEL = "CNN-BiGRU-Attention-BiGRU-based"

# Custom Deep Hybrid models
CNN_ATTENTION_LSTM_MODEL = "CNN-Attention-LSTM-based"
CNN_ATTENTION_GRU_MODEL = "CNN-Attention-GRU-based"
CNN_ATTENTION_BiLSTM_MODEL = "CNN-Attention-BiLSTM-based"
CNN_ATTENTION_BiGRU_MODEL = "CNN-Attention-BiGRU-based"

# Custom Mode Deep Hybrid models
CNN_ATTENTION_LSTM_ATTENTION_MODEL = "CNN-Attention-LSTM-Attention-based"
CNN_ATTENTION_GRU_ATTENTION_MODEL = "CNN-Attention-GRU-Attention-based"
CNN_ATTENTION_BiLSTM_ATTENTION_MODEL = "CNN-Attention-BiLSTM-Attention-based"
CNN_ATTENTION_BiGRU_ATTENTION_MODEL = "CNN-Attention-BiGRU-Attention-based"

'''---------------------------------------------------------------------------'''
# Define saving paths
SAVING_MODEL_DIR = "../models/"
SAVING_METRIC_DIR = "metrics/"
SAVING_PREDICTION_DIR = "predictions/"
SAVING_LOSS_DIR = "losses/"
SAVING_METRICS_PATH = "metrics/evaluation_metrics.json"
SAVING_LOSSES_PATH = "losses/models_losses.json"

# Define dirs
DATASET_FEATURES_PATH = f"input/data_features.json"
ELECTRICITY_DATASET_PATH = f"input/electricity/household_power_consumption.txt"
GOLD_DATASET_PATH = f"input/gold/GoldPrice.csv"
AIR_DATASET_PATH = f"input/air/AirQualityUCI.csv"
OUTPUT_PATH = f"output-cpu/"
BASE_PATH = f'./'
CHECK_PATH = "checks/"
CHECK_HYPERBAND = "hyperband/"
HYPERBAND_PATH = "hyperband/"
LOG_FILE = './logs/cpu/time_serie_cpu.log'

# Define model names as variables
EPOCHS = 1
N_TRIAL = 1
SEEDER = 2024
LOOK_BACKS = [7]
FORECAST_PERIODS = [1]
ELECTRICITY = 'electricity'
APARTMENT = 'apartment'


def is_running_on_server():
    # We assume that the server is Linux
    return os.uname().sysname == 'Linux'


logger = logging.getLogger(__name__)
if is_running_on_server():
    logger.info("The code is running on a server.")
    EPOCHS = 100
    N_TRIAL = 500
    LOOK_BACKS = [14, 30, 60]
    FORECAST_PERIODS = [1, 2, 3, 6, 7]

    BASE_PATH = '/home/23r9802_chen/messou/TimeSerieForecasting/'
    DATASET_FEATURES_PATH = "/home/23r9802_chen/messou/TimeSerieForecasting/input/data_features.json"
    ELECTRICITY_DATASET_PATH = "/home/23r9802_chen/messou/TimeSerieForecasting/input/electricity/household_power_consumption.txt"
    GOLD_DATASET_PATH = "/home/23r9802_chen/messou/TimeSerieForecasting/input/gold/GoldPrice.csv"
    AIR_DATASET_PATH = "/home/23r9802_chen/messou/TimeSerieForecasting/input/air/AirQualityUCI.csv"
    OUTPUT_PATH = "/output-gpu/"
    LOG_FILE = f"./logs/gpu/time_serie_gpu"
else:
    logger.info("The code is running locally.")
