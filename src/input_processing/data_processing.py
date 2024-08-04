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
from sklearn.preprocessing import MinMaxScaler

from utils.constants import (
    DATASET_FEATURES_PATH,
    ELECTRICITY_DATASET_PATH,
    ELECTRICITY,
    APARTMENT,
    HOUSE,
    WATER,
    APARTMENT_DATASET_PATH,
    HOUSE_DATASET_PATH,
    UNI_OR_MULTI_VARIATE,
    PREPROCESSED_ELECTRICITY_DATASET_PATH,
    PREPROCESSED_HOUSE_DATASET_PATH,
    PREPROCESSED_ENERGY_DATASET_PATH,
    PREPROCESSED_APARTMENT_DATASET_PATH,
    PREPROCESSED_WATER_DATASET_PATH,
    WATER_DATASET_PATH,
)
from utils.constants import ENERGY, ENERGY_DATASET_PATH, SIMPLE_OR_AUGMENTED
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
        X.append(dataset[i : (i + look_back), :])
        Y.append(dataset[(i + look_back) : (i + look_back + forecast_period), 0])
    return np.array(X), np.array(Y)


def save_preprocessed_dataset(scaled_dataset, url):
    if url == ELECTRICITY:
        scaled_dataset.to_csv(
            PREPROCESSED_ELECTRICITY_DATASET_PATH + "_preprocessed.csv", index=False
        )
    elif url == APARTMENT:
        scaled_dataset.to_csv(
            PREPROCESSED_APARTMENT_DATASET_PATH + "_preprocessed.csv", index=False
        )
    elif url == HOUSE:
        scaled_dataset.to_csv(
            PREPROCESSED_HOUSE_DATASET_PATH + "_preprocessed.csv", index=False
        )
    elif url == ENERGY:
        scaled_dataset.to_csv(
            PREPROCESSED_ENERGY_DATASET_PATH + "_preprocessed.csv", index=False
        )
    elif url == WATER:
        scaled_dataset.to_csv(
            PREPROCESSED_WATER_DATASET_PATH + "_preprocessed.csv", index=False
        )


def load_preprocessed_dataset(url="electricity"):
    if url == ELECTRICITY:
        dataset = pd.read_csv(
            PREPROCESSED_ELECTRICITY_DATASET_PATH + "_preprocessed.csv", na_values=["?"]
        )
    elif url == APARTMENT:
        dataset = pd.read_csv(
            PREPROCESSED_APARTMENT_DATASET_PATH + "_preprocessed.csv", na_values=["?"]
        )
    elif url == HOUSE:
        dataset = pd.read_csv(
            PREPROCESSED_HOUSE_DATASET_PATH + "_preprocessed.csv", na_values=["?"]
        )
    elif url == ENERGY:
        dataset = pd.read_csv(
            PREPROCESSED_ENERGY_DATASET_PATH + "_preprocessed.csv", na_values=["?"]
        )
    elif url == WATER:
        dataset = pd.read_csv(
            PREPROCESSED_WATER_DATASET_PATH + "_preprocessed.csv", na_values=["?"]
        )

    return dataset


def tuning_load_dataset(dataset_type="electricity", period="D"):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv(ELECTRICITY_DATASET_PATH, sep=";", na_values=["?"])
        dataset["datetime"] = pd.to_datetime(
            dataset["Date"] + " " + dataset["Time"], format="%d/%m/%Y %H:%M:%S"
        )
        dataset.drop(["Date", "Time"], axis=1, inplace=True)

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        selected_features = selected_features + ["day_of_week", "month"]

        # Resample data and drop NA values that might be created during resampling
        data = (
            df.set_index("datetime")[selected_features].resample(period).mean().dropna()
        )

        # Separate features and target variable
        # features = data.values
        # target = data[selected_features[0]].values.reshape(-1, 1)

    return data


def default_load_dataset(dataset_type="electricity", period="D"):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv(ELECTRICITY_DATASET_PATH, sep=";", na_values=["?"])
        dataset["datetime"] = pd.to_datetime(
            dataset["Date"] + " " + dataset["Time"], format="%d/%m/%Y %H:%M:%S"
        )
        dataset.drop(["Date", "Time"], axis=1, inplace=True)

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month

        # Assuming 'selected_features' is obtained from elsewhere
        if UNI_OR_MULTI_VARIATE == "multivariate":
            selected_features = selected_features + ["day_of_week", "month"]
        else:
            selected_features = selected_features[0]

        # Resample data and drop NA values that might be created during resampling
        data = (
            df.set_index("datetime")[selected_features].resample(period).mean().dropna()
        )
    elif dataset_type == ENERGY:
        dataset = pd.read_csv(ENERGY_DATASET_PATH, na_values=["?"])
        # Convert Date/Time column to datetime format with specific format
        dataset["date"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d %H:%M:%S")

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        if UNI_OR_MULTI_VARIATE == "multivariate":
            selected_features = selected_features + ["day_of_week", "month"]
        else:
            selected_features = selected_features[:, 0]

        # Resample data and drop NA values that might be created during resampling
        data = df.set_index("date")[selected_features].resample(period).mean().dropna()
    elif dataset_type == WATER:
        dataset = pd.read_csv(WATER_DATASET_PATH, na_values=["?"])
        # Convert Date/Time column to datetime format with specific format
        dataset["time"] = pd.to_datetime(dataset["time"], format="%Y/%m/%d %H:%M")

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df["day_of_week"] = df["time"].dt.dayofweek
        df["month"] = df["time"].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        if UNI_OR_MULTI_VARIATE == "multivariate":
            selected_features = selected_features + ["day_of_week", "month"]
        else:
            selected_features = selected_features[:, 0]

        # Resample data and drop NA values that might be created during resampling
        data = df.set_index("time")[selected_features].resample(period).mean().dropna()
    else:
        if dataset_type == APARTMENT:
            dataset = pd.read_csv(APARTMENT_DATASET_PATH, na_values=["?"])
            # Convert Date/Time column to datetime format with specific format
            dataset["Date/Time"] = pd.to_datetime(
                dataset["Date/Time"], format="%Y-%m-%d %H:%M:%S"
            )
        elif dataset_type == HOUSE:
            dataset = pd.read_csv(HOUSE_DATASET_PATH, na_values=["?"])
            # Convert Date/Time column to datetime format with specific format
            dataset["Date/Time"] = pd.to_datetime(
                dataset["Date/Time"], format="%Y/%m/%d %H:%M"
            )
        elif dataset_type == ENERGY:
            dataset = pd.read_csv(WATER_DATASET_PATH, na_values=["?"])
            # Convert Date/Time column to datetime format with specific format
            dataset["date"] = pd.to_datetime(dataset["date"], format="%Y/%m/%d %H:%M")
        elif dataset_type == WATER:
            dataset = pd.read_csv(WATER_DATASET_PATH, na_values=["?"])
            dataset["time"] = pd.to_datetime(dataset["time"], format="%Y/%m/%d %H:%M")
        else:
            raise ValueError("Cannot load dataset type {}".format(dataset_type))

        df = fill_missing_data(dataset, meth=0)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        # Extract time features
        df["day_of_week"] = df["Date/Time"].dt.dayofweek
        df["month"] = df["Date/Time"].dt.month

        # Assuming 'selected_features' is obtained from elsewhere in your code
        if UNI_OR_MULTI_VARIATE == "multivariate":
            selected_features = selected_features + ["day_of_week", "month"]
        else:
            selected_features = selected_features[:, 0]

        # Resample data and drop NA values that might be created during resampling
        data = (
            df.set_index("Date/Time")[selected_features]
            .resample(period)
            .mean()
            .dropna()
        )

    # Separate features and target variable
    # features = data.values
    # target = data[selected_features[0]].values.reshape(-1, 1)

    return data


def default_preprocess_and_split_dataset(url, period, look_back, forecast_period):
    # Load dataset and fill missing values
    loaded_dataset = default_load_dataset(url, period)

    # Combine features and target variable
    # dataset = np.concatenate((features, target), axis=1)

    if SIMPLE_OR_AUGMENTED == "augmented":
        processed_data = robust_data_augmentation(loaded_dataset, url=url)

    # Normalize entire dataset

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(processed_data)
    print(processed_data.shape)
    if SIMPLE_OR_AUGMENTED == "augmented":
        dataset_tosave = pd.DataFrame(scaled_dataset, columns=loaded_dataset.columns)
        save_preprocessed_dataset(dataset_tosave, url)

    # Split dataset into input sequences (X) and target sequences (y)
    X, y = create_dataset(scaled_dataset, look_back, forecast_period)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


def validation_split_dataset(url, look_back, forecast_period):
    scaler = MinMaxScaler(feature_range=(0, 1))
    preprocessed_dataset = load_preprocessed_dataset(url)
    X, y = create_dataset(preprocessed_dataset.values, look_back, forecast_period)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


def tuning_preprocess_and_split_dataset(url, period, look_back, forecast_period):
    preprocessed_dataset = load_preprocessed_dataset(url)
    if SIMPLE_OR_AUGMENTED == "augmented":
        dataset = robust_data_augmentation(preprocessed_dataset, url=url)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset)

    X, y = create_dataset(scaled_dataset, look_back, forecast_period)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler
