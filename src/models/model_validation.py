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

import torch

from hyperparameter_tuning.build_best_model import train_model
from input_processing.data_processing import validation_split_dataset, default_preprocess_and_split_dataset
from models.model_selection import ModelSelection
from output_processing.custom_functions import (evaluate_model, plot_evaluation_metrics, save_evaluation_metrics,
                                                plot_losses, save_losses, make_predictions, plot_multi_step_predictions,
                                                save_predicted_values)
from utils.constants import (
    SAVING_METRIC_DIR, SAVING_LOSS_DIR, BASE_PATH,
    OUTPUT_PATH, SAVING_PREDICTION_DIR, HYPERBAND_PATH, ELECTRICITY, PERIOD, SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE
)
from utils.file_loader import read_best_params

logger = logging.getLogger(__name__)


class ComprehensiveModelValidator:
    def __init__(self, series_types, look_backs, forecast_periods, model_types):
        self.series_types = series_types
        self.look_backs = look_backs
        self.forecast_periods = forecast_periods
        self.model_types = model_types

    def train_best_model(self, model, x_train, y_train, x_val, y_val, device):
        _, train_history = train_model(model, torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device),
                                       torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device), device)
        return train_history

    def build_and_train_models(self):
        serie_type = self.series_types
        for model_type in self.model_types:
            for look_back_day in self.look_backs:
                for forecast_day in self.forecast_periods:
                    for period in PERIOD:
                        logger.info(
                            f"Training with series_type={serie_type} | look_back={look_back_day} | forecast_period={forecast_day}")
                        if SIMPLE_OR_AUGMENTED == "simple":
                            x_train, x_test, y_train, y_test, scaler = default_preprocess_and_split_dataset(serie_type,
                                                                                                            'D',
                                                                                                        look_back_day,
                                                                                                            forecast_day)
                        else:
                            x_train, x_test, y_train, y_test, scaler = validation_split_dataset(serie_type,
                                                                                                look_back_day,
                                                                                                        forecast_day)
                        self.process_model_type(model_type, serie_type, look_back_day, forecast_day, period, x_train,
                                                x_test,
                                                y_train, y_test, scaler)

    def process_model_type(self, model_type, series_type, look_back_day, forecast_day, period, x_train, x_test, y_train,
                           y_test,
                           scaler):
        logger.info(f'\n**************************************************************\n'
                    f'** {model_type}\n'
                    f'**************************************************************')

        loading_path_best_params = f"{BASE_PATH + OUTPUT_PATH + ELECTRICITY}/{HYPERBAND_PATH}{model_type}/best_params.json"
        logger.info(f"LOADING  PATH 📌📌📌  {loading_path_best_params}  📌📌📌")

        # LOAD PYTORCH MODEL WITH BEST HYPERPARAMETERS
        best_params = read_best_params(loading_path_best_params, model_type)
        try:
            best_params = read_best_params(loading_path_best_params, model_type)
            if best_params is None:
                print(f"No best parameters found for {model_type}. Continuing to the next model type.")
        except KeyError as e:
            print(e)
            print(f"Skipping {model_type} and continuing to the next model type.")
        logger.info("Best params loaded: %s", best_params)

        model_selector = ModelSelection(X_train=x_train, output_dim=forecast_day, model_type=model_type)
        model = model_selector.select_model(best_params=best_params, trial=None)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_history = self.train_best_model(model, x_train, y_train, x_test, y_test, device)
        self.evaluate_and_save_results(model, x_test, y_test, scaler, train_history, model_type, series_type,
                                       look_back_day,
                                       forecast_day, period)

    def evaluate_and_save_results(self, model, x_test, y_test, scaler, train_history, model_type, series_type,
                                  look_back_day,
                                  forecast_day, period):
        saving_path_metric = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_METRIC_DIR}/{model_type}/{SIMPLE_OR_AUGMENTED}/"
        saving_path_loss = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_LOSS_DIR}/{model_type}/{SIMPLE_OR_AUGMENTED}/"
        saving_path_prediction = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_PREDICTION_DIR}/{model_type}/{SIMPLE_OR_AUGMENTED}/"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Making predictions
        model.eval()
        with torch.no_grad():
            predictions = model(torch.Tensor(x_test).to(device))
            predictions = predictions.cpu().numpy()

        # Calculating evaluation metrics
        mse, mae, rmse, mape = evaluate_model(y_test,
                                              predictions)

        # Logging the metrics for this model
        logger.info(
            f"Model: {model_type}-{SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE}, Look Back: {look_back_day}, Forecast Day: {forecast_day},  Period: {period}, MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")

        # Saving the evaluation metrics - ensure the saving path is defined and accessible
        plot_evaluation_metrics(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, mse, mae, rmse, mape, model_type,
                                look_back_day, forecast_day,
                                period,
                                saving_path_metric)
        save_evaluation_metrics(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, mse, mae, rmse, mape, model_type,
                                look_back_day, forecast_day,
                                period,
                                saving_path_metric)
        # Plotting and saving loss curves
        plot_losses(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, train_history, model_type, look_back_day, forecast_day,
                    period,
                    saving_path_loss)
        save_losses(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, train_history, model_type, look_back_day, forecast_day,
                    period,
                    saving_path_loss)

        # Plotting and saving predictions
        testPredict, testOutput = make_predictions(model, x_test, y_test, scaler)
        save_predicted_values(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, testOutput, testPredict, model_type,
                              look_back_day, forecast_day, period, saving_path_prediction)
        # plot_single_prediction(SIMPLE_OR_AUGMENTED,UNI_OR_MULTI_VARIATE,  testPredict,testOutput, model_type, look_back_day, forecast_day,period,saving_path_prediction)
        plot_multi_step_predictions(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, testPredict, testOutput, model_type,
                                    series_type,
                                    look_back_day,
                                    forecast_day, period,
                                    saving_path_prediction)

# class ComprehensiveModelValidator:
#     def __init__(self, series_types, look_backs, forecast_periods, model_types):
#         self.series_types = series_types
#         self.look_backs = look_backs
#         self.forecast_periods = forecast_periods
#         self.model_types = model_types
#
#     def train_best_model(self, model, x_train, y_train, x_val, y_val, device):
#         _, train_history = train_model(model, torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device),
#                                        torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device), device)
#         return train_history
#
#     def build_and_train_models(self):
#         serie_type = self.series_types
#         for model_type in self.model_types:
#             for look_back_day in self.look_backs:
#                 for forecast_day in self.forecast_periods:
#                     for period in PERIOD:
#                         logger.info(
#                             f"Training with series_type={serie_type} | look_back={look_back_day} | forecast_period={forecast_day}")
#                         x_train, x_val, x_test, y_train, y_val, y_test, scaler = validation_preprocess_and_split_dataset(
#                             serie_type,
#                             period,
#                             look_back_day,
#                             forecast_day)
#                         self.process_model_type(model_type, serie_type, look_back_day, forecast_day, period, x_train,
#                                                 x_val, x_test, y_train, y_val, y_test, scaler)
#
#     def process_model_type(self, model_type, series_type, look_back_day, forecast_day, period, x_train, x_val, x_test,
#                            y_train, y_val, y_test, scaler):
#         logger.info(f'\n**************************************************************\n'
#                     f'** {model_type}\n'
#                     f'**************************************************************')
#
#         loading_path_best_params = f"{BASE_PATH + OUTPUT_PATH + ELECTRICITY}/{HYPERBAND_PATH}{model_type}/best_params.json"
#         logger.info(f"LOADING  PATH 📌📌📌  {loading_path_best_params}  📌📌📌")
#
#         # LOAD PYTORCH MODEL WITH BEST HYPERPARAMETERS
#         best_params = read_best_params(loading_path_best_params, model_type)
#         try:
#             best_params = read_best_params(loading_path_best_params, model_type)
#             if best_params is None:
#                 print(f"No best parameters found for {model_type}. Continuing to the next model type.")
#         except KeyError as e:
#             print(e)
#             print(f"Skipping {model_type} and continuing to the next model type.")
#         logger.info("Best params loaded: %s", best_params)
#
#         model_selector = ModelSelection(X_train=x_train, output_dim=forecast_day, model_type=model_type)
#         model = model_selector.select_model(best_params=best_params, trial=None)
#
#         # Move model to GPU if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#
#         train_history = self.train_best_model(model, x_train, y_train, x_val, y_val, device)
#         self.evaluate_and_save_results(model, x_test, y_test, scaler, train_history, model_type, series_type,
#                                        look_back_day,
#                                        forecast_day, period)
#
#     def evaluate_and_save_results(self, model, x_test, y_test, scaler, train_history, model_type, series_type,
#                                   look_back_day,
#                                   forecast_day, period):
#         saving_path_metric = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_METRIC_DIR}/{model_type}/{SIMPLE_OR_AUGMENTED}/"
#         saving_path_loss = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_LOSS_DIR}/{model_type}/{SIMPLE_OR_AUGMENTED}/"
#         saving_path_prediction = f"{BASE_PATH + OUTPUT_PATH + series_type}/{SAVING_PREDICTION_DIR}/{model_type}/{SIMPLE_OR_AUGMENTED}/"
#
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Making predictions
#         model.eval()
#         with torch.no_grad():
#             predictions = model(torch.Tensor(x_test).to(device))
#             predictions = predictions.cpu().numpy()
#
#         # Calculating evaluation metrics
#         mse, mae, rmse, mape = evaluate_model(y_test,
#                                               predictions)
#
#         # Logging the metrics for this model
#         logger.info(
#             f"Model: {model_type}-{SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE}, Look Back: {look_back_day}, Forecast Day: {forecast_day},  Period: {period}, MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
#
#         # Saving the evaluation metrics - ensure the saving path is defined and accessible
#         plot_evaluation_metrics(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, mse, mae, rmse, mape, model_type,
#                                 look_back_day, forecast_day,
#                                 period,
#                                 saving_path_metric)
#         save_evaluation_metrics(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, mse, mae, rmse, mape, model_type,
#                                 look_back_day, forecast_day,
#                                 period,
#                                 saving_path_metric)
#
#         # Plotting and saving loss curves
#         plot_losses(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, train_history, model_type, look_back_day, forecast_day,
#                     period,
#                     saving_path_loss)
#         save_losses(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, train_history, model_type, look_back_day, forecast_day,
#                     period,
#                     saving_path_loss)
#         # Plotting and saving predictions
#         testPredict, testOutput = make_predictions(model, x_test, y_test, scaler)
#         save_predicted_values(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, testOutput, testPredict, model_type,
#                               look_back_day, forecast_day, period, saving_path_prediction)
#         # plot_single_prediction(SIMPLE_OR_AUGMENTED,UNI_OR_MULTI_VARIATE,  testPredict,testOutput, model_type, look_back_day, forecast_day,period,saving_path_prediction)
#         plot_multi_step_predictions(SIMPLE_OR_AUGMENTED, UNI_OR_MULTI_VARIATE, testPredict, testOutput, model_type,
#                                     series_type,
#                                     look_back_day,
#                                     forecast_day, period,
#                                     saving_path_prediction)
