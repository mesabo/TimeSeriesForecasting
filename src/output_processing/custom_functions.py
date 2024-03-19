#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:49:39 2024

@author: mesabo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error)
import json
import os
import torch


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


def evaluate_model(testY, testPredict):
    mse = round(mean_squared_error(testY, testPredict), 6)
    mae = round(mean_absolute_error(testY, testPredict), 6)
    rmse = round(mean_squared_error(testY, testPredict, squared=False), 6)
    mape = mean_absolute_percentage_error(testY, testPredict) * 100
    mape = round(mape, 6)
    print(f"[-----MODEL METRICS-----]\n")
    print(f"[-----MSE: {mse}-----]\n")
    print(f"[-----MAE: {mae}-----]\n")
    print(f"[-----RMSE: {rmse}-----]\n")
    print(f"[-----MAPE: {mape}-----]\n")
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


def plot_losses(history, model, save_path=None):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_name = f'{model}_evaluation_metrics.png'
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path)

    plt.show()


def plot_evaluation_metrics(mse, mae, rmse, mape, model, save_path=None):
    metrics = ['MSE', 'MAE', 'RMSE']
    values = [mse, mae, rmse]

    plt.bar(metrics, values, color=['orange', 'limegreen', 'steelblue'])
    plt.title(f'{model} - Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_name = f'{model}_evaluation_metrics.png'
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path)

    plt.show()


def plot_predictions(predicted, actual, model, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(actual[:int(len(actual / 3)), 0], label='Actual')
    plt.plot(predicted[:int(len(predicted / 3)), 0], label='Predicted')
    plt.xlabel('Period')
    plt.ylabel('Global Active Power')
    plt.title(f'{model} - Actual vs Predicted')
    plt.legend()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_name = f'{model}_prediction.png'
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path)

    plt.show()


def save_evaluation_metrics(saving_path, model_type, mse, mae, rmse, mape):
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    if os.path.exists(saving_path):
        with open(saving_path, 'r') as file:
            evaluation_data = json.load(file)
    else:
        evaluation_data = {}

    evaluation_data[model_type] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

    # Save the updated data back to the file
    with open(saving_path, 'w') as file:
        json.dump(evaluation_data, file, indent=2)


def save_loss_to_txt(saving_path, model_type, history):
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    if os.path.exists(saving_path):
        with open(saving_path, 'r') as file:
            loss_data = json.load(file)
    else:
        loss_data = {}

    loss_data[model_type] = {
        'training_loss': history.history['loss'],
        'validation_loss': history.history['val_loss']
    }

    with open(saving_path, 'w') as file:
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
