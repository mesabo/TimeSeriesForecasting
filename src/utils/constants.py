#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:35:52 2024

@author: mesabo
"""

# Define model names as variables
EPOCH = 100
BATCH_SIZE = 64
SEEDER = 2024
PARAMS_GRID = {'batch_size': [16, 32, 64, 128, 256, 512]}
LOOK_BACKS = [7, 10, 14, 30]
FORECAST_PERIODS = [1, 2, 3, 6, 7]
ELECTRICITY = 'electricity'
WATER = 'water'
WIND = 'wind'
GOLD = 'gold'
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

# Define saving paths
SAVING_MODEL_DIR = "../models/"
SAVING_METRIC_DIR = "metrics/"
SAVING_PREDICTION_DIR = "predictions/"
SAVING_LOSS_DIR = "losses/"
SAVING_METRICS_PATH = "metrics/evaluation_metrics.json"
SAVING_LOSSES_PATH = "losses/models_losses.json"

# Define dataset paths
DATASET_FEATURES_PATH = "./input/data_features.json"
ELECTRICITY_DATASET_PATH = "./input/electricity/household_power_consumption.txt"
GOLD_DATASET_PATH = "../input/gold/GoldPrice.csv"
AIR_DATASET_PATH = "../input/air/AirQualityUCI.csv"

BASE_PATH = "./output/"
CHECK_PATH = "checks/"
CHECK_HYPERBAND = "hyperband/"
HYPERBAND_PATH = "hyperband/"
# CHECK_HYPERBAND_PATH = "hyperband/best_params.json"
