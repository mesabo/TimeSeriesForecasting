# main.py
# model_tuner.py

import torch
import torch.nn as nn
import optuna
import json
import time
from src.hyperparameter_tuning.model_training import train_model
from src.hyperparameter_tuning.hyper_models import BuildCNNLSTMAttentionModel
from src.input_processing.data_processing import preprocess_augment_and_split_dataset
from src.utils.constants import (
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
    ELECTRICITY, LOOK_BACKS, FORECAST_PERIODS
)
from src.hyperparameter_tuning.model_tuner import ModelTuner

import torch
import optuna
import json
from src.hyperparameter_tuning.model_training import train_model
from src.hyperparameter_tuning.hyper_models import BuildCNNLSTMAttentionModel
from src.hyperparameter_tuning.model_tuner_study import model_tuner_and_study
import optuna


def build_best_model(study, X_train, y_train, X_val, y_val, input_dim, forecast_days):
    # Best hyperparameters
    best_params = study.best_trial.params

    # PyTorch model with best hyperparameters
    model = BuildCNNLSTMAttentionModel(input_dim=input_dim, num_cnn_layers=best_params['num_cnn_layers'],
                                       num_lstm_layers=best_params['num_lstm_layers'], filters=best_params['filters'],
                                       kernel_size=best_params['kernel_size'],
                                       lstm_units=best_params['lstm_units'], dropout=best_params['dropout'],
                                       activation=best_params['activation'], output_dim=forecast_days, trial=None)

    print(model)

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

    print("RMSE of the best model:", rmse)

    # Save output
    output_file_path = f"./output/sample.json"
    with open(output_file_path, "w") as output_file:
        json.dump(study.best_trial.params, output_file)

    print("Output saved to:", output_file_path)

    return model, device, rmse


def main():
    series = [ELECTRICITY]
    model_types = [CNN_LSTM_ATTENTION_MODEL]

    look_backs = [7]
    forecast_periods = [1, 3]

    # Create ModelTuner instance and Optuna study
    model_tuner, study, best_params = model_tuner_and_study(look_backs, forecast_periods, model_types, series)

    ''''# Build best model
    # Load and split dataset
    X_train, X_val, y_train, y_val, _ = preprocess_augment_and_split_dataset(ELECTRICITY, 'D', look_backs[0],
                                                                             forecast_periods[0])
    input_dim = X_train.shape[2]
    #model, device, rmse = build_best_model(study, X_train, y_train, X_val, y_val, input_dim, forecast_periods[0])
    '''


if __name__ == "__main__":
    main()
