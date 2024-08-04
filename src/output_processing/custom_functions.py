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

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error,
                             root_mean_squared_error)

from utils.constants import (ELECTRICITY, APARTMENT, ENERGY, HOUSE)

logger = logging.getLogger(__name__)


def format_duration(duration):
    if duration < 60:
        return f"{duration:.2f} s"
    elif duration < 3600:
        return f"{duration / 60:.2f} min"
    elif duration < 86400:
        return f"{duration / 3600:.2f} h"
    elif duration < 31536000:
        return f"{duration / 86400:.2f} d"
    else:
        return f"{duration / 31536000:.2f} year"


def make_predictions(model, testX, testY, scaler):
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        testX_tensor = torch.tensor(testX, dtype=torch.float32).to(device)
        tensorPredict = model(testX_tensor).cpu().numpy()

    return tensorPredict, testY


def evaluate_model(testY, testPredict):
    mse = round(mean_squared_error(testY, testPredict), 6)
    mae = round(mean_absolute_error(testY, testPredict), 6)
    rmse = round(root_mean_squared_error(testY, testPredict), 6)
    mape = mean_absolute_percentage_error(testY, testPredict)
    mape = round(mape, 6)
    logger.info("[-----MODEL METRICS-----]\n")
    logger.info(f"[-----MSE: {mse}-----]\n")
    logger.info(f"[-----MAE: {mae}-----]\n")
    logger.info(f"[-----RMSE: {rmse}-----]\n")
    logger.info(f"[-----MAPE: {mape}-----]\n")
    return mse, mae, rmse, mape


def plot_evaluation_metrics(simple_or_augmented, uni_or_multi_variate, mse, mae, rmse, mape, model_type, look_back,
                            forecast_day, period, save_path=None):
    metrics = ['MSE', 'MAE', 'RMSE', ]
    values = [mse, mae, rmse, ]

    plt.bar(metrics, values, color=['limegreen', 'steelblue', 'purple', ])
    plt.title(f'{model_type} - {uni_or_multi_variate} Metrics - ({look_back} x {period} lookback)')
    plt.xlabel('Metric')
    plt.ylabel('Value')

    if save_path:
        file_name = f'{look_back}_{forecast_day}_{period}_evaluation_metrics_{simple_or_augmented}_{uni_or_multi_variate}.png'
        file_path = os.path.join(save_path, 'image', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)

    # plt.show()
    plt.clf()


def plot_losses(simple_or_augmented, uni_or_multi_variate, history, model_type, look_back, forecast_day, period,
                save_path=None):
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} - {uni_or_multi_variate} Loss - ({look_back} x {period} lookback)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        file_name = f'{look_back}_{forecast_day}_{period}_evaluation_metrics_{simple_or_augmented}_{uni_or_multi_variate}.png'
        file_dir = os.path.join(save_path, 'image')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)
        plt.savefig(file_path)

    # plt.show()
    plt.clf()


def plot_single_prediction(simple_or_augmented, uni_or_multi_variate, predicted, actual, model_type, series_type,
                           look_back,
                           forecast_day, period, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(actual[:, -1], label='Actual')
    for i in len(predicted):
        plt.plot(predicted[i][:, -1], label=model_type[i])
    plt.xlabel(f'Timestamp resolution ({period})')
    if series_type == ELECTRICITY:
        plt.ylabel('Global Active Power (Kw)')
    elif series_type == APARTMENT:
        plt.ylabel('Facility (Kw)')
    elif series_type == ENERGY:
        plt.ylabel('Appliances (Kw)')
    elif series_type == HOUSE:
        plt.ylabel('Power (Kw)')
    plt.title(f'{model_type} - {uni_or_multi_variate} - Actual vs Predicted - ({look_back} x {period} lookback) ')
    plt.legend()
    if save_path:
        file_name = f'{look_back}_{forecast_day}_{period}_single_prediction_{simple_or_augmented}_{uni_or_multi_variate}.png'
        file_dir = os.path.join(save_path, 'image')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)
        plt.savefig(file_path)

    # plt.show()
    plt.clf()


def plot_multi_step_predictions(simple_or_augmented, uni_or_multi_variate, predicted, actual, model_type, series_type,
                                look_back,
                                forecast_days, period,
                                save_path=None):
    plt.figure(figsize=(10, 8))

    # Plot actual values for all forecast days
    plt.plot(actual[-400:, 0], label='Actual')

    # Plot predicted values with different colors for each step ahead
    for i in range(predicted.shape[1]):
        if i == 0:
            plt.plot(predicted[-400:, i], label=f'{i + 1} step ahead')
        else:
            plt.plot(predicted[-400:, i], label=f'{i + 1} steps ahead')

    plt.xlabel(f'Timestamp resolution ({period})')
    if series_type == ELECTRICITY:
        plt.ylabel('Global Active Power (Kw)')
    elif series_type == APARTMENT:
        plt.ylabel('Facility (Kw)')
    elif series_type == ENERGY:
        plt.ylabel('Appliances (Kw)')
    elif series_type == HOUSE:
        plt.ylabel('Power (Kw)')
    plt.title(f'{model_type} - Actual vs Predicted - ({look_back} x {period} lookback) - {uni_or_multi_variate}')
    plt.legend()

    if save_path:
        file_name = f'{look_back}_{forecast_days}_{period}_prediction_{simple_or_augmented}_{uni_or_multi_variate}.png'
        file_dir = os.path.join(save_path, 'image')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)
        plt.savefig(file_path)

    # plt.show()
    plt.clf()


def save_predicted_values(simple_or_augmented, uni_or_multi_variate, actual, predicted, model_type, look_back,
                          forecast_day, period,
                          save_path=None):
    # Define the file name and directory
    file_name = f'{look_back}_{forecast_day}_{period}_prediction_{simple_or_augmented}_{uni_or_multi_variate}.json'
    file_dir = os.path.join(save_path, 'doc') if save_path else 'doc'
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, file_name)

    # # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            loss_data = json.load(file)
    else:
        loss_data = {}

    # Convert numpy arrays to lists for JSON serialization
    actual_list = actual.tolist() if isinstance(actual, np.ndarray) else actual
    predicted_list = predicted.tolist() if isinstance(predicted, np.ndarray) else predicted

    # Update the loss_data dictionary
    loss_data[model_type] = {
        'actual': actual_list,
        'predicted': predicted_list
    }

    # Save the updated dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(loss_data, file, indent=2)


def save_evaluation_metrics(simple_or_augmented, uni_or_multi_variate, mse, mae, rmse, mape, model_type, look_back,
                            forecast_day, period,
                            save_path=None):
    file_name = f'{look_back}_{forecast_day}_{period}_evaluation_metrics_{simple_or_augmented}_{uni_or_multi_variate}.json'
    file_path = os.path.join(save_path, 'doc', file_name)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            evaluation_data = json.load(file)
    else:
        evaluation_data = {}

    evaluation_data[model_type] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

    with open(file_path, 'w') as file:
        json.dump(evaluation_data, file, indent=2)


def save_losses(simple_or_augmented, uni_or_multi_variate, history, model_type, look_back, forecast_day, period,
                save_path=None):
    file_name = f'{look_back}_{forecast_day}_{period}_evaluation_losses_{simple_or_augmented}_{uni_or_multi_variate}.json'
    file_dir = os.path.join(save_path, 'doc')
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            loss_data = json.load(file)
    else:
        loss_data = {}

    loss_data[model_type] = {
        'training_loss': history['train_loss'],
        'validation_loss': history['val_loss']
    }

    with open(file_path, 'w') as file:
        json.dump(loss_data, file, indent=2)


def save_best_params(saving_path, model_type, best_hps, total_time):
    directory = os.path.dirname(saving_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(saving_path):
        with open(saving_path, 'r') as file:
            evaluation_data = json.load(file)
    else:
        evaluation_data = {}

    # Update or add the hyperparameters for the model type
    evaluation_data[model_type] = best_hps
    evaluation_data[model_type]['processing_time'] = format_duration(total_time)

    # Save the updated data back to the file
    with open(saving_path, 'w') as file:
        json.dump(evaluation_data, file, indent=2)


def save_trained_model(model, path):
    torch.save(model.state_dict(), path)


def load_trained_model(path, device=torch.device("cpu")):
    model = torch.load(path, map_location=device)
    model.eval()  # Set model to evaluation mode after loading
    return model
