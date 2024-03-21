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
                             ELECTRICITY)
from utils.file_loader import read_features


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


def time_warping(series, sigma=0.2):
    n = len(series)
    time_stretching = np.cumsum(np.random.randn(n) * sigma)
    time_stretching -= time_stretching.min()
    time_stretching /= time_stretching.max()
    new_time = np.arange(n)
    warped_time = new_time + time_stretching * (n - 1)
    warped_series = interp1d(warped_time, series, bounds_error=False, fill_value="extrapolate")(new_time)
    return warped_series


def load_dataset(dataset_type='electricity', period='D'):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv(ELECTRICITY_DATASET_PATH, sep=';', na_values=['?'])
        dataset['datetime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H:%M:%S')
        dataset.drop(['Date', 'Time'], axis=1, inplace=True)

        df = fill_missing_data(dataset, meth=2)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        data = df.set_index('datetime')[selected_features].resample(period).mean().dropna()
        # Separate features and target variable
        features = data.drop(columns=selected_features[0]).values
        target = data[selected_features[0]].values.reshape(-1, 1)

    return features, target


def preprocess_augment_and_split_dataset(url, period, look_back, forecast_period):
    # Load dataset and fill missing values
    features, target = load_dataset(url, period)

    # Normalize features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    # Normalize target variable
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    # Combine scaled features and target variable
    scaled_dataset = np.concatenate((scaled_features, scaled_target), axis=1)
    # Apply time warping to the features (excluding the target variable)
    warped_features = np.apply_along_axis(time_warping, axis=0, arr=scaled_dataset[:, :-1])

    # Combine warped features with target variable
    warped_dataset = np.concatenate((warped_features, scaled_dataset[:, -1].reshape(-1, 1)), axis=1)

    # Split dataset into input sequences (X) and target sequences (y)
    X, y = create_dataset(warped_dataset, look_back, forecast_period)

    # Split dataset into train and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size

    X_train, X_test = X[:train_size], X[-test_size:]
    y_train, y_test = y[:train_size], y[-test_size:]

    return X_train, X_test, y_train, y_test, scaler_target


def preprocess_and_split_dataset(url, period, look_back, forecast_period):
    # Load dataset and fill missing values
    features, target = load_dataset(url, period)

    # Normalize features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    # Normalize target variable
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    # Combine scaled features and target variable
    scaled_dataset = np.concatenate((scaled_features, scaled_target), axis=1)

    # Split dataset into input sequences (X) and target sequences (y)
    X, y = create_dataset(scaled_dataset, look_back, forecast_period)

    # Split dataset into train and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size

    X_train, X_test = X[:train_size], X[-test_size:]
    y_train, y_test = y[:train_size], y[-test_size:]

    return X_train, X_test, y_train, y_test, scaler_target
