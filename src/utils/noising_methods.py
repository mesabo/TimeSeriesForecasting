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


def add_noise(data):
    """
    Add normal distribution noise with varying standard deviations to the data.

    Parameters:
    - data: numpy array, input data concatenated with target

    Returns:
    - numpy array, data with added noise
    """
    noise_levels = np.random.uniform(0.1, 1.0, size=data.shape[0])  # Sample noise levels from a range
    noised_data = np.copy(data)
    for i, level in enumerate(noise_levels):
        noise = np.random.normal(0, level, size=data.shape[1])  # Ensure noise shape matches the data shape
        noised_data[i] += noise
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
        x = np.arange(data_with_target.shape[1] - 1)  # Exclude target from x
        y = permuted_data[i, :-1]  # Exclude target from y
        if len(x) < 4:  # Ensure there are enough data points for cubic spline
            continue  # Skip spline fitting if there are not enough data points
        spline = UnivariateSpline(x, y, k=3, s=20)  # Adjust smoothing factor (s) as needed
        noise = np.random.normal(0, 0.1, data_with_target.shape[1] - 1)  # To be adjusted as needed
        permuted_data[i, :-1] = spline(x) + noise
    return permuted_data



def scale_data(data):
    """
    Scale the amplitude of the data.

    Parameters:
    - data: numpy array, input data concatenated with target

    Returns:
    - numpy array, scaled data
    """
    scaling_factors = np.random.uniform(0.5, 2.0, size=data.shape[0])  # Sample scaling factors from a range
    scaled_data = np.copy(data)
    for i, factor in enumerate(scaling_factors):
        scaled_data[i, :-1] *= factor  # Exclude target from scaling
    return scaled_data


def warp_data(data):
    """
    Apply random warping to the data.

    Parameters:
    - data: numpy array, input data concatenated with target

    Returns:
    - numpy array, warped data
    """
    warped_data = np.copy(data)
    for i in range(data.shape[0]):
        num_control_points = np.random.randint(2, 10)  # Adjust the number of control points as needed
        control_points = np.linspace(0, data.shape[1] - 2,
                                     num_control_points)  # Exclude target from control_points
        warp_factors = np.random.uniform(0.8, 1.2, num_control_points)  # Adjust warp factors as needed
        warped_data[i, :-1] = np.interp(np.arange(data.shape[1] - 1), control_points,
                                        control_points * warp_factors)
    return warped_data


def robust_data_augmentation(features, target):
    """
    Apply augmentation methods to the concatenated dataset while maintaining consistency between features and target.

    Parameters:
    - features: numpy array, input features
    - target: numpy array, target values corresponding to the features

    Returns:
    - numpy array, augmented features
    - numpy array, target values corresponding to the augmented features
    """
    # Concatenate features and target

    # Apply augmentation methods
    augmented_features = add_noise(features)
    augmented_features = permute(augmented_features)
    augmented_features = scale_data(augmented_features)
    augmented_features = warp_data(augmented_features)

    augmented_target = add_noise(target)
    augmented_target = permute(augmented_target)
    augmented_target = scale_data(augmented_target)
    augmented_target = warp_data(augmented_target)

    return augmented_features, augmented_target
