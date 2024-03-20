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
    model.eval()
    testX_tensor = torch.Tensor(testX)

    with torch.no_grad():
        testPredict = model(testX_tensor).cpu().numpy()

    testPredict = scaler.inverse_transform(testPredict)
    testOutput = scaler.inverse_transform(testY)

    return testPredict, testOutput


def evaluate_model(testY, testPredict):
    mse = round(mean_squared_error(testY, testPredict), 6)
    mae = round(mean_absolute_error(testY, testPredict), 6)
    rmse = round(root_mean_squared_error(testY, testPredict), 6)
    mape = mean_absolute_percentage_error(testY, testPredict) * 100
    mape = round(mape, 6)
    logger.info("[-----MODEL METRICS-----]\n")
    logger.info(f"[-----MSE: {mse}-----]\n")
    logger.info(f"[-----MAE: {mae}-----]\n")
    logger.info(f"[-----RMSE: {rmse}-----]\n")
    logger.info(f"[-----MAPE: {mape}-----]\n")
    return mse, mae, rmse, mape


def predict_next_x_days(model, X_new, days=7):
    predictions = []

    # Iterate over the next days
    for i in range(days):
        prediction = model.predict(X_new)

        predictions.append(prediction)
        # Update X_new for the next iteration
        # Shift the values by one day and append the new prediction
        X_new = np.roll(X_new, -1, axis=1)
        X_new[-1] = prediction[0]

    predictions = np.array(predictions)

    return predictions


def plot_predictions(predicted, actual, model_type, look_back, forecast_day, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(actual[:int(len(actual / 3)), 0], label='Actual')
    plt.plot(predicted[:int(len(predicted / 3)), 0], label='Predicted')
    plt.xlabel('Period')
    plt.ylabel('Global Active Power')
    plt.title(f'{model_type} - Actual vs Predicted')
    plt.legend()
    if save_path:
        file_name = f'{look_back}_{forecast_day}_prediction.png'
        file_path = os.path.join(save_path, 'IMAGE', file_name)
        plt.savefig(file_path)
    plt.show()


def plot_evaluation_metrics(mse, mae, rmse, mape, model_type, look_back, forecast_day, save_path=None):
    metrics = ['MSE', 'MAE', 'RMSE',
               # 'MAPE',
               ]
    values = [mse, mae, rmse,
              # mape
              ]

    plt.bar(metrics, values, color=['limegreen', 'steelblue', 'purple',
                                    # 'orange',
                                    ])
    plt.title(f'{model_type} - Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')

    if save_path:
        file_name = f'{look_back}_{forecast_day}_evaluation_metrics.png'
        file_path = os.path.join(save_path, 'image', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)

    plt.show()


def plot_losses(history, model_type, look_back, forecast_day, save_path=None):
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        file_name = f'{look_back}_{forecast_day}_evaluation_metrics.png'
        file_dir = os.path.join(save_path, 'image')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)
        plt.savefig(file_path)

    plt.show()


def plot_predictions(predicted, actual, model_type, look_back, forecast_day, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(actual[:int(len(actual / 3)), 0], label='Actual')
    plt.plot(predicted[:int(len(predicted / 3)), 0], label='Predicted')
    plt.xlabel('Period')
    plt.ylabel('Global Active Power')
    plt.title(f'{model_type} - Actual vs Predicted')
    plt.legend()
    if save_path:
        file_name = f'{look_back}_{forecast_day}_prediction.png'
        file_dir = os.path.join(save_path, 'image')
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, file_name)
        plt.savefig(file_path)
    plt.show()


def save_evaluation_metrics(mse, mae, rmse, mape, model_type, look_back, forecast_day, save_path=None):
    file_name = f'{look_back}_{forecast_day}_evaluation_metrics.json'
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


def save_loss_to_json(history, model_type, look_back, forecast_day, save_path=None):
    file_name = f'{look_back}_{forecast_day}_evaluation_losses.json'
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
