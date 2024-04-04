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
# input_processing.py

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

from utils.constants import (DATASET_FEATURES_PATH, ELECTRICITY_DATASET_PATH,
                             ELECTRICITY, APARTMENT, APARTMENT_DATASET_PATH,
                             HOUSE_DATASET_PATH, HOUSE)
from utils.file_loader import read_features
from utils.noising_methods import robust_data_augmentation


def fill_missing_data(data, meth=2):
    if meth == 1:
        # 2. Imputation with Simple Statistics
        # Replace missing values with the mean for numeric columns
        data.fillna(data.mean(), inplace=True)
    elif meth == 2:
        # 3. Forward or Backward Fill (Time Series Data)
        data.sort_values(by="datetime", inplace=True)
        data.ffill(inplace=True)  # Forward fill
    elif meth == 3:
        # 4. Interpolation
        # Linear interpolation for numeric columns
        data.interpolate(method="linear", inplace=True)
    else:
        # 1. Dropping Rows or Columns
        # Drop rows with any missing values
        data = data.dropna()

    return data


def create_dataset(dataset, look_back, forecast_period):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_period + 1):
        X.append(dataset[i:(i + look_back), :])
        Y.append(
            dataset[(i + look_back):(i + look_back + forecast_period), 0])
    return np.array(X), np.array(Y)


def tuning_load_dataset(dataset_type='electricity', period='D'):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv(ELECTRICITY_DATASET_PATH, sep=';', na_values=['?'])
        dataset['datetime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H:%M:%S')
        dataset.drop(['Date', 'Time'], axis=1, inplace=True)

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        selected_features = selected_features + ['hour', 'day_of_week', 'month']

        # Resample data and drop NA values that might be created during resampling
        data = df.set_index('datetime')[selected_features].resample(period).mean().dropna()

        # Separate features and target variable
        # features = data.values
        # target = data[selected_features[0]].values.reshape(-1, 1)

    return data


def default_load_dataset(dataset_type='electricity', period='D'):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv(ELECTRICITY_DATASET_PATH, sep=';', na_values=['?'])
        dataset['datetime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H:%M:%S')
        dataset.drop(['Date', 'Time'], axis=1, inplace=True)

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        selected_features = selected_features + ['hour', 'day_of_week', 'month']

        # Resample data and drop NA values that might be created during resampling
        data = df.set_index('datetime')[selected_features].resample(period).mean().dropna()

    else:
        if dataset_type == APARTMENT:
            dataset = pd.read_csv(APARTMENT_DATASET_PATH, na_values=['?'])
            # Convert Date/Time column to datetime format with specific format
            dataset['Date/Time'] = pd.to_datetime(dataset['Date/Time'], format='%Y-%m-%d %H:%M:%S')
        elif dataset_type == HOUSE:
            dataset = pd.read_csv(HOUSE_DATASET_PATH, na_values=['?'])
            # Convert Date/Time column to datetime format with specific format
            dataset['Date/Time'] = pd.to_datetime(dataset['Date/Time'], format='%Y/%m/%d %H:%M')
        else:
            raise ValueError('Cannot load dataset type {}'.format(dataset_type))

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df['hour'] = df['Date/Time'].dt.hour
        df['day_of_week'] = df['Date/Time'].dt.dayofweek
        df['month'] = df['Date/Time'].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        selected_features = selected_features + ['hour', 'day_of_week', 'month']

        # Resample data and drop NA values that might be created during resampling
        data = df.set_index('Date/Time')[selected_features].resample(period).mean().dropna()

    # Separate features and target variable
    # features = data.values
    # target = data[selected_features[0]].values.reshape(-1, 1)

    return data


def default_preprocess_and_split_dataset(url, period, look_back, forecast_period):
    # Load dataset and fill missing values
    dataset = default_load_dataset(url, period)

    # Combine features and target variable
    # dataset = np.concatenate((features, target), axis=1)

    # Normalize entire dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)

    # Split dataset into input sequences (X) and target sequences (y)
    X, y = create_dataset(scaled_dataset, look_back, forecast_period)

    # Split dataset into train and test sets
    if len(X) > 1000 and (period == 'd' or period == 'D' or period == '1d'):
        train_size = int(len(X) - 365)
    else:
        train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


def tuning_preprocess_and_split_dataset(url, period, look_back, forecast_period):
    # Load dataset and fill missing values
    dataset = tuning_load_dataset(url, period)
    # Combine features and target variable
    # dataset = np.concatenate((features, target), axis=1)

    # augmented_dataset = robust_data_augmentation(dataset)

    # Normalize entire dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)

    # Split dataset into input sequences (X) and target sequences (y)
    X, y = create_dataset(scaled_dataset, look_back, forecast_period)

    # Split dataset into train and test sets
    if len(X) > 1000 and (period == 'd' or period == 'D' or period == '1d'):
        train_size = int(len(X) - 365)
    else:
        train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler
