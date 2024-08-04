#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/03/2024
🚀 Welcome to the Awesome Python Script 🚀

User: mesabo
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab

"""
import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.constants import ELECTRICITY, ELECTRICITY_DATASET_PATH, DATASET_FEATURES_PATH, ENERGY, ENERGY_DATASET_PATH, \
    APARTMENT, APARTMENT_DATASET_PATH, HOUSE, HOUSE_DATASET_PATH
from utils.file_loader import read_features


def load_dataset(dataset_type):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv('../../' + ELECTRICITY_DATASET_PATH, sep=';', na_values=['?'])
        dataset['datetime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H:%M:%S')
        dataset.drop(['Date', 'Time'], axis=1, inplace=True)
    elif dataset_type == ENERGY:
        dataset = pd.read_csv('../../' + ENERGY_DATASET_PATH, na_values=['?'])
        dataset['date'] = pd.to_datetime(dataset['date'], format='%Y-%m-%d %H:%M:%S')
    elif dataset_type == APARTMENT:
        dataset_path = APARTMENT_DATASET_PATH if dataset_type == APARTMENT else HOUSE_DATASET_PATH
        dataset = pd.read_csv('../../' + dataset_path, na_values=['?'])
        dataset['Date/Time'] = pd.to_datetime(dataset['Date/Time'], format='%Y-%m-%d %H:%M:%S')
    elif dataset_type == HOUSE:
        dataset_path = APARTMENT_DATASET_PATH if dataset_type == APARTMENT else HOUSE_DATASET_PATH
        dataset = pd.read_csv('../../' + dataset_path, na_values=['?'])
        dataset['Date/Time'] = pd.to_datetime(dataset['Date/Time'], format='%Y/%m/%d %H:%M')
    else:
        raise ValueError('Cannot load dataset type {}'.format(dataset_type))

    selected_features = read_features('../../' + DATASET_FEATURES_PATH, dataset_type)
    data = \
        dataset.set_index(
            'datetime' if dataset_type == ELECTRICITY else 'date' if dataset_type == ENERGY else 'Date/Time')[
            selected_features]

    return data


def plot_histograms(datasets, labels):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for ax, dataset, label in zip(axes.flatten(), datasets, labels):
        for feature in dataset.columns:
            ax.hist(dataset[feature], bins=30, alpha=0.5, label=feature)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label} Histogram (Overlay)')
        ax.legend()
    plt.tight_layout()
    # Save plot as PNG
    filename = f'../../output-cpu/analysis/histogram_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.eps'
    plt.savefig(filename, dpi=50, rasterized=True)
    plt.show()

def plot_cdfs(datasets, labels):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    for ax, dataset, label in zip(axes, datasets, labels):
        for feature in dataset.columns:
            sorted_data = np.sort(dataset[feature])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=feature)
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{label} CDF')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 1])  # Set y-axis limits to [0, 1] for cumulative probability
    plt.tight_layout()

    # Save plot as EPS (change filename as needed)
    filename = f'../../output-cpu/analysis/tes_cdf_{datetime.datetime.now(datetime.UTC)}.eps'
    plt.savefig(filename, format='eps')  # Specify format as 'eps'

    # Optional: Show the plot (comment out for headless environments)
    plt.show()




def read_json(file_path):
    """
    Reads a JSON file and returns its content.

    :param file_path: Path to the JSON file.
    :return: Data from JSON file or None if file not found.
    """
    try:
        with open(file_path, "r") as file:
            data_features = json.load(file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

    return data_features


def plot_multi_step_predictions(actual, predicted):
    """
    Plots multi-step predictions comparing actual vs predicted values.

    :param actual: 2D list of actual values.
    :param predicted: 2D list of predicted values.
    """
    plt.figure(figsize=(10, 8))

    # Convert lists to numpy arrays for easier indexing
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Plot actual values for all forecast days
    plt.plot(actual[-100:, 0], label='Actual', marker='o')

    # Plot predicted values with different colors for each step ahead
    # for i in range(predicted.shape[1]):
    plt.plot(predicted[-100:, 0], label=f'Simple Dataset prediction', marker='o')
    plt.plot(predicted[-100:, 1], label=f'-Augmented Dataset prediction', marker='o')

    # plt.plot(predicted[-100:, 1], label=f'-prediction 1', marker='o')
    # plt.plot(predicted[-100:, 2], label=f'-prediction 2', marker='o')
    # plt.plot(predicted[-100:, 3], label=f'-prediction 3', marker='o')
    # plt.plot(predicted[-100:, 4], label=f'-prediction 4', marker='o')
    # plt.plot(predicted[-100:, 5], label=f'-prediction 5', marker='o')

    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Multi-step ahead - Actual vs Predicted')
    plt.legend()

    # Save the plot with a unique filename based on the current datetime
    # filename = f"./multi_step_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    # plt.savefig(filename)

    plt.show()

def main():
    # np.random.seed(42)
    # electricity = load_dataset(ELECTRICITY)
    # energy = load_dataset(ENERGY)
    # apartment = load_dataset(APARTMENT)
    # house = load_dataset(HOUSE)
    #
    # datasets, labels = [electricity, apartment, house, energy], [ELECTRICITY, APARTMENT, HOUSE, ENERGY]
    #
    # # Plotting histograms
    # plot_histograms(datasets, labels)

    # Plotting CDFs
    # plot_cdfs(datasets, labels)

    # Generate predicted values for the model
    model_name = "CNN-Attention-BiLSTM-based"
    filename = f'../../output-cpu/analysis/simple_augmented.json'
    pred_data = read_json(filename)
    plot_multi_step_predictions(actual=pred_data['actual'], predicted=pred_data['predicted'])


if __name__ == "__main__":
    main()
