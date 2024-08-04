#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/03/2024
🚀 Welcome to the Awesome Python Script 🚀

User: mesabo
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab

"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from utils.constants import ELECTRICITY


def add_noise(data):
    """
    Add normal distribution noise with varying standard deviations to the data.

    Parameters:
    - data: numpy array, input data

    Returns:
    - numpy array, data with added noise
    """
    noise_levels = np.random.uniform(0.1, 1.0, size=data.shape[0])
    noised_data = np.copy(data)
    for i, level in enumerate(noise_levels):
        noise = np.random.normal(0, level, size=data.shape[1] - 3)  # Exclude target from noise
        noised_data[i, 1:-2] += noise  # Exclude target from noise addition
    return noised_data


def permute(data_with_target):
    """
    Apply random curves augmentation to the data.

    Parameters:
    - data_with_target: numpy array, input data concatenated with target

    Returns:
    - numpy array, permuted data
    """
    permuted_data = np.copy(data_with_target)
    for i in range(data_with_target.shape[0]):
        x = np.arange(data_with_target.shape[1] - 3)
        y = permuted_data[i, 1:-2]
        if len(x) < 4:
            continue
        spline = UnivariateSpline(x, y, k=3, s=20)
        noise = np.random.normal(0, 0.1, len(x))
        permuted_data[i, 1:-2] = spline(x) + noise
    return permuted_data


def scale_data(data):
    """
    Scale the amplitude of the data.

    Parameters:
    - data: numpy array, input data

    Returns:
    - numpy array, scaled data
    """
    scaling_factors = np.random.uniform(0.5, 2.0, size=data.shape[0])  # Sample scaling factors from a range
    scaled_data = np.copy(data)
    for i, factor in enumerate(scaling_factors):
        scaled_data[i, 1:-2] *= factor  # Exclude target from scaling
    return scaled_data


def warp_data(data):
    """
    Apply random warping to the data.

    Parameters:
    - data: numpy array, input data

    Returns:
    - numpy array, warped data
    """
    warped_data = np.copy(data)
    for i in range(data.shape[0]):
        num_control_points = np.random.randint(2, 10)  # To adjust the number of control points as needed
        control_points = np.linspace(1.25, data.shape[1] - 3,
                                     num_control_points)  # To exclude target from control_points
        warp_factors = np.random.uniform(0.8, 1.2, num_control_points)  # Adjust warp factors as needed
        warped_data[i, 1:-2] = np.interp(np.arange(data.shape[1] - 3), control_points,
                                       control_points * warp_factors)
    return warped_data


def robust_data_augmentation(dataset, url=None):
    """
    Apply augmentation methods to the concatenated dataset while maintaining consistency between features and target.

    Parameters:
    - features: numpy array, input features
    - target: numpy array, target values corresponding to the features

    Returns:
    - numpy array, augmented features
    - numpy array, target values corresponding to the augmented features
    """
    loop = 1
    data = np.copy(dataset)
    datasets = [data]
    if url is not None and url != ELECTRICITY:
        loop = 3
    for i in range(0, loop):
        augmented_data = add_noise(data)
        augmented_data = permute(augmented_data)
        augmented_data = scale_data(augmented_data)
        augmented_data = warp_data(augmented_data)
        datasets.append(augmented_data)

    # Concatenate the augmented dataset with itself to double its size
    augmented_dataset = np.concatenate(datasets, axis=0)

    return augmented_dataset
