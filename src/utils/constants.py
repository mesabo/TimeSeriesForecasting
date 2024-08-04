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
import platform

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
SAVING_EVALUATIONS_DIR = "evaluations/"
SAVING_METRICS_PATH = "metrics/evaluation_metrics.json"
SAVING_LOSSES_PATH = "losses/models_losses.json"

# Define dirs
DATASET_FEATURES_PATH = f"input/data_features.json"
ELECTRICITY_DATASET_PATH = f"input/electricity/household_power_consumption.txt"
HOUSE_DATASET_PATH = f"input/house/WHE.csv"
APARTMENT_DATASET_PATH = f"input/apartment/MidriseApartment_SAN_FRANCISCO.csv"
ENERGY_DATASET_PATH = f"input/energy/energy.csv"
WATER_DATASET_PATH = f"input/water/water.csv"

PREPROCESSED_ELECTRICITY_DATASET_PATH = f"input/electricity/household_power_consumption"
PREPROCESSED_HOUSE_DATASET_PATH = f"input/house/WHE"
PREPROCESSED_APARTMENT_DATASET_PATH = f"input/apartment/MidriseApartment_SAN_FRANCISCO"
PREPROCESSED_ENERGY_DATASET_PATH = f"input/energy/energy"
PREPROCESSED_WATER_DATASET_PATH = f"input/water/water"

OUTPUT_PATH = f"output-cpu/"
BASE_PATH = f'./'
CHECK_PATH = "checks/"
CHECK_HYPERBAND = "hyperband/"
HYPERBAND_PATH = "hyperband/"
LOG_FILE = './logs/cpu/'
UNI_OR_MULTI_VARIATE = 'multivariate'
SIMPLE_OR_AUGMENTED = 'augmented'
# Define model names as variables
SEEDER = 2024

EPOCHS = 200
N_TRIAL = 50
LOOK_BACKS = [7]
FORECAST_PERIODS = [1, 2, 3, 4, 5, 6, 7]
PERIOD = ['1d']
ELECTRICITY = 'electricity'
HOUSE = 'house'
APARTMENT = 'apartment'
ENERGY = 'energy'
WATER = 'water'


def is_running_on_server():
    # We assume that the server is Linux
    return platform.system() == 'Linux'


logger = logging.getLogger(__name__)
if is_running_on_server():
    logger.info("The code is running on a server.")
    EPOCHS = 200
    N_TRIAL = 50
    LOOK_BACKS = [7]
    FORECAST_PERIODS = [1, 2, 3, 4, 5, 6]

    BASE_PATH = '/home/23r9802_chen/messou/TimeSeriesForecasting/'
    DATASET_FEATURES_PATH = "/home/23r9802_chen/messou/TimeSeriesForecasting/input/data_features.json"
    ELECTRICITY_DATASET_PATH = "/home/23r9802_chen/messou/TimeSeriesForecasting/input/electricity/household_power_consumption.txt"
    HOUSE_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/house/WHE.csv"
    APARTMENT_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/apartment/MidriseApartment_SAN_FRANCISCO.csv"
    ENERGY_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/energy/energy.csv"
    WATER_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/water/water.csv"

    PREPROCESSED_ELECTRICITY_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/electricity/household_power_consumption"
    PREPROCESSED_HOUSE_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/house/WHE"
    PREPROCESSED_APARTMENT_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/apartment/MidriseApartment_SAN_FRANCISCO"
    PREPROCESSED_ENERGY_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/energy/energy"
    PREPROCESSED_WATER_DATASET_PATH = f"/home/23r9802_chen/messou/TimeSeriesForecasting/input/water/water"

    OUTPUT_PATH = "/output-gpu/"
    LOG_FILE = f"./logs/gpu/"
else:
    logger.info("The code is running locally.")
