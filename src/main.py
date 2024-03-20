# main.py

import os
import torch.nn as nn
import torch
import json
from hyperparameter_tuning.hyper_models import BuildCNNLSTMAttentionModel
from hyperparameter_tuning.model_training import train_model
from hyperparameter_tuning.model_tuner_study import model_tuner_and_study
import optuna
import logging
import socket
import psutil
import platform

from input_processing.data_processing import preprocess_augment_and_split_dataset
# from hyperparameter_tuning.model_training import train_model
# from hyperparameter_tuning.hyper_models import BuildCNNLSTMAttentionModel
# from input_processing.data_processing import preprocess_augment_and_split_dataset

from utils.constants import (
    LSTM_MODEL, GRU_MODEL, CNN_MODEL, BiLSTM_MODEL, BiGRU_MODEL,
    LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL, CNN_ATTENTION_MODEL,
    BiLSTM_ATTENTION_MODEL, BiGRU_ATTENTION_MODEL,
    CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
    CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_ATTENTION_MODEL, CNN_ATTENTION_GRU_ATTENTION_MODEL,
    CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_MODEL, CNN_ATTENTION_GRU_MODEL,
    CNN_ATTENTION_BiLSTM_MODEL, CNN_ATTENTION_BiGRU_MODEL,
    SAVING_MODEL_DIR, SAVING_METRIC_DIR, SAVING_LOSS_DIR, BASE_PATH,
    SAVING_PREDICTION_DIR, SAVING_METRICS_PATH, SAVING_LOSSES_PATH, SEEDER,
    HYPERBAND_PATH, DATASET_FEATURES_PATH, ELECTRICITY_DATASET_PATH,
    ELECTRICITY, LOOK_BACKS, FORECAST_PERIODS, OUTPUT_PATH
)


# from hyperparameter_tuning.model_tuner import ModelTuner


def build_best_model(study, X_train, y_train, X_val, y_val, input_dim, forecast_days):
    # Best hyperparameters
    best_params = study.best_trial.params

    # PyTorch model with best hyperparameters
    model = BuildCNNLSTMAttentionModel(input_dim=input_dim, num_cnn_layers=best_params['num_cnn_layers'],
                                       num_lstm_layers=best_params['num_lstm_layers'], filters=best_params['filters'],
                                       kernel_size=best_params['kernel_size'],
                                       lstm_units=best_params['lstm_units'], dropout=best_params['dropout'],
                                       activation=best_params['activation'], output_dim=forecast_days, trial=None)

    logger.info(model)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training the best model on the full training set
    val_loss = train_model(model, torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device),
                           torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device),
                           optuna.trial.FixedTrial(best_params), device)

    # Calculate RMSE on test set
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_output = model(torch.Tensor(X_val).to(device))
        test_loss = criterion(test_output, torch.Tensor(y_val).to(device))
        rmse = torch.sqrt(test_loss).item()

    logger.info(f"RMSE of the best model: {rmse}", )

    # Save output
    output_file_path = f"{OUTPUT_PATH}sample.json"
    with open(output_file_path, "w") as output_file:
        json.dump(study.best_trial.params, output_file)

    logger.info(f"Output saved to:{output_file_path}")

    return model, device, rmse


def main():
    series = [ELECTRICITY]
    model_types = [CNN_LSTM_ATTENTION_MODEL]

    look_backs = LOOK_BACKS  # [7]
    forecast_periods = FORECAST_PERIODS  # [3]

    # Create ModelTuner instance and Optuna study
    model_tuner, study, best_params = model_tuner_and_study(look_backs, forecast_periods, model_types, series)

    # Build best model
    # Load and split dataset
    X_train, X_val, y_train, y_val, _ = preprocess_augment_and_split_dataset(ELECTRICITY, 'D', look_backs[0],
                                                                             forecast_periods[0])
    input_dim = X_train.shape[2]
    model, device, rmse = build_best_model(study, X_train, y_train, X_val, y_val, input_dim, forecast_periods[0])


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
cpu_info, cpu_count = None,None
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

if __name__ == "__main__":
    # Call the function to trigger logging
    some_function()
    current_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"CURRENT PATH IS: {current_dir}")
    main()
