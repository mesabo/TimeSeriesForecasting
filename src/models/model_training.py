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

import optuna
import torch

from hyperparameter_tuning.build_best_model import train_model
from hyperparameter_tuning.hyper_models import BuildCNNLSTMAttentionModel
from input_processing.data_processing import preprocess_and_split_dataset
from output_processing.custom_functions import (evaluate_model, plot_evaluation_metrics, save_evaluation_metrics,
                                                plot_losses, save_loss_to_json, make_predictions, plot_predictions)
from utils.constants import (
    SAVING_METRIC_DIR, SAVING_LOSS_DIR, BASE_PATH,
    OUTPUT_PATH, SAVING_PREDICTION_DIR
)

logger = logging.getLogger(__name__)


def train_best_model(model, x_train, y_train, x_val, y_val, best_params, device):
    _, train_history = train_model(model, torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device),
                                   torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device),
                                   optuna.trial.FixedTrial(best_params), device)
    return train_history


def build_best_model(study, x_train, y_train, x_val, y_val, scaler, input_dim, look_back, forecast_day, model_type,
                     series_type):
    logger.info(f'**************************************************************\n'
                f'**                {model_type}                **\n'
                f'**************************************************************')
    # ex: ./output/electricity/metrics//CNN-LSTM-Attention-based/
    saving_path_metric = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_METRIC_DIR}/{model_type}/"
    saving_path_loss = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_LOSS_DIR}/{model_type}/"
    saving_path_prediction = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_PREDICTION_DIR}/{model_type}/"
    logger.info(f"BASE PATH 📌📌📌  {saving_path_metric}  📌📌📌")

    # Best hyperparameters
    best_params = study.best_trial.params

    # PyTorch model with best hyperparameters
    model = BuildCNNLSTMAttentionModel(input_dim=input_dim, num_cnn_layers=best_params['num_cnn_layers'],
                                       num_lstm_layers=best_params['num_lstm_layers'], filters=best_params['filters'],
                                       kernel_size=best_params['kernel_size'],
                                       lstm_units=best_params['lstm_units'], dropout=best_params['dropout'],
                                       activation=best_params['activation'], output_dim=forecast_day, trial=None)

    logger.info(model)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the best model
    train_history = train_best_model(model, x_train, y_train, x_val, y_val, best_params, device)

    # Convert model predictions to numpy arrays
    with torch.no_grad():
        val_output = model(torch.Tensor(x_val).to(device))
        val_predictions = val_output.cpu().numpy()

    X_train_pred, X_test_pred, y_train_pred, y_test_pred, scaler = preprocess_and_split_dataset(series_type, 'D',
                                                                                                look_back, forecast_day)

    # Evaluate model
    mse, mae, rmse, mape = evaluate_model(y_val, val_predictions)
    plot_evaluation_metrics(mse, mae, rmse, mape, model_type, look_back, forecast_day, saving_path_metric)
    save_evaluation_metrics(mse, mae, rmse, mape, model_type, look_back, forecast_day, saving_path_metric)

    # Plot losses
    plot_losses(train_history, model_type, look_back, forecast_day, saving_path_loss)
    save_loss_to_json(train_history, model_type, look_back, forecast_day, saving_path_loss)

    testPredict, testOutput = make_predictions(model, X_test_pred, y_test_pred, scaler)
    plot_predictions(testPredict, testOutput, model_type, look_back, forecast_day, saving_path_prediction)
