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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=';', parse_dates={'DateTime': ['Date', 'Time']}, infer_datetime_format=True,
                     index_col='DateTime')
    return df


# Explore the dataset
def explore_dataset(data):
    print("Dataset information:")
    print(data.info())
    print("\nDescriptive statistics:")
    print(data.describe())
    print("\nHead of the dataset:")
    print(data.head())


# Plot time series data
def plot_time_series(data, start_date=None, end_date=None):
    if start_date and end_date:
        data = data.loc[start_date:end_date]
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Global_active_power'], color='blue', label='Global Active Power')
    plt.title('Global Active Power Over Time')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.show()


# Visualize distribution of features
def visualize_distribution(data):
    plt.figure(figsize=(14, 6))
    data.hist(bins=50, figsize=(16, 10))
    plt.suptitle('Histograms of Features', x=0.5, y=1.02, ha='center', fontsize='x-large')
    plt.show()


# Correlation analysis
def correlation_analysis(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    plt.matshow(corr_matrix, cmap='viridis', fignum=1)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.show()


def visualization_distribution(data):
    # Distribution of the target variables
    sns.histplot(data=data, x='Global_active_power', bins=15, kde=True)
    plt.show()


def main():
    # Load dataset
    file_path = '../../input/electricity/household_power_consumption.txt'
    data = load_dataset(file_path)

    # Explore dataset
    explore_dataset(data)

    # Plot time series data
    # plot_time_series(data, start_date='2007-01-01', end_date='2007-01-02')

    # Visualize distribution of features
    # visualize_distribution(data)

    # Distribution
    visualization_distribution(data)


if __name__ == "__main__":
    main()
