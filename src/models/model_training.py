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
from input_processing.data_processing import preprocess_and_split_dataset, preprocess_augment_and_split_dataset
from output_processing.custom_functions import (evaluate_model, plot_evaluation_metrics, save_evaluation_metrics,
                                                plot_losses, save_loss_to_json, make_predictions, plot_predictions)
from utils.constants import (
    SAVING_METRIC_DIR, SAVING_LOSS_DIR, BASE_PATH,
    OUTPUT_PATH, SAVING_PREDICTION_DIR, HYPERBAND_PATH
)
from utils.file_loader import read_best_params

logger = logging.getLogger(__name__)


def train_best_model(model, x_train, y_train, x_val, y_val, best_params, device):
    _, train_history = train_model(model, torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device),
                                   torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device),
                                   optuna.trial.FixedTrial(best_params), device)
    return train_history


def build_best_model(look_backs, forecast_periods, model_types, series_type):
    for _ser in series_type:
        for look_back_day in look_backs:
            for forecast_day in forecast_periods:
                logger.info(
                    f"Training with series_type={_ser} | look_back={look_back_day} | forecast_period={forecast_day}")
                x_train, x_val, y_train, y_val, _ = preprocess_augment_and_split_dataset(_ser, 'D',
                                                                                         look_back_day,
                                                                                         forecast_day)
                for model_type in model_types:
                    logger.info(f'**************************************************************\n'
                                f'** {model_type}\n'
                                f'**************************************************************')

                    saving_path_metric = f"{BASE_PATH + OUTPUT_PATH + _ser}/{SAVING_METRIC_DIR}/{model_type}/"
                    saving_path_loss = f"{BASE_PATH + OUTPUT_PATH + _ser}/{SAVING_LOSS_DIR}/{model_type}/"
                    saving_path_prediction = f"{BASE_PATH + OUTPUT_PATH + _ser}/{SAVING_PREDICTION_DIR}/{model_type}/"
                    loading_path_best_params = f"{BASE_PATH + OUTPUT_PATH + _ser}/{HYPERBAND_PATH}{model_type}/{look_back_day}_{forecast_day}_best_params.json"
                    logger.info(f"LOADING  PATH 📌📌📌  {loading_path_best_params}  📌📌📌")

                    # LOAD PYTORCH MODEL WITH BEST HYPERPARAMETERS
                    best_params = read_best_params(loading_path_best_params, model_type)
                    logger.info("Best params loaded: %s", best_params)
                    input_dim = x_train.shape[2]

                    model = BuildCNNLSTMAttentionModel(input_dim=input_dim,
                                                       best_params=best_params,
                                                       output_dim=forecast_day,
                                                       trial=None)

                    # Move model to GPU if available
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)

                    # TRAIN THE BEST MODEL
                    train_history = train_best_model(model, x_train, y_train, x_val, y_val, best_params, device)
                    # Convert model predictions to numpy arrays
                    with torch.no_grad():
                        val_output = model(torch.Tensor(x_val).to(device))
                        val_predictions = val_output.cpu().numpy()

                    # EVALUATE MODEL
                    X_train_pred, X_test_pred, y_train_pred, y_test_pred, scaler = preprocess_and_split_dataset(
                        _ser, 'D',
                        look_back_day, forecast_day)

                    mse, mae, rmse, mape = evaluate_model(y_val, val_predictions)
                    plot_evaluation_metrics(mse, mae, rmse, mape, model_type, look_back_day, forecast_day,
                                            saving_path_metric)
                    save_evaluation_metrics(mse, mae, rmse, mape, model_type, look_back_day, forecast_day,
                                            saving_path_metric)

                    # Plot losses
                    plot_losses(train_history, model_type, look_back_day, forecast_day, saving_path_loss)
                    save_loss_to_json(train_history, model_type, look_back_day, forecast_day, saving_path_loss)

                    testPredict, testOutput = make_predictions(model, X_test_pred, y_test_pred, scaler)
                    plot_predictions(testPredict, testOutput, model_type, look_back_day, forecast_day,
                                     saving_path_prediction)
