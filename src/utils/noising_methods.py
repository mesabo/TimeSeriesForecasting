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
    - data: numpy array, input data

    Returns:
    - numpy array, data with added noise
    """
    noise_levels = np.random.uniform(0.1, 1.0, size=data.shape[0])  # Sample noise levels from a range
    noised_data = np.copy(data)
    for i, level in enumerate(noise_levels):
        noise = np.random.normal(0, level, data.shape[1])
        noised_data[i] += noise
    return noised_data

def permute(data):
    """
    Apply random curves augmentation to the data.

    Parameters:
    - data: numpy array, input data

    Returns:
    - numpy array, permuted data
    """
    permuted_data = np.copy(data)
    for i in range(data.shape[0]):
        x = np.arange(data.shape[1])
        y = permuted_data[i]
        spline = UnivariateSpline(x, y, k=3)
        noise = np.random.normal(0, 0.1, data.shape[1])  # Adjust noise level as needed
        permuted_data[i] = spline(x) + noise
    return permuted_data

def scale_data(data):
    """
    Scale the amplitude of the data.

    Parameters:
    - data: numpy array, input data

    Returns:
    - numpy array, scaled data
    """
    scaling_factors = np.random.uniform(0.5, 2.0, size=data.shape[0])
    scaled_data = np.copy(data)
    for i, factor in enumerate(scaling_factors):
        scaled_data[i] *= factor
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
        num_control_points = np.random.randint(2, 10)
        control_points = np.linspace(0, data.shape[1] - 1, num_control_points)
        warp_factors = np.random.uniform(0.8, 1.2, num_control_points)
        warped_data[i] = np.interp(np.arange(data.shape[1]), control_points, control_points * warp_factors)
    return warped_data

def robust_data_augmentation(dataset):
    """
    Apply robust data augmentation techniques to the dataset.

    Parameters:
    - dataset: numpy array, input dataset

    Returns:
    - numpy array, augmented data
    """
    # Apply noise
    noised_data = add_noise(dataset)

    # Apply permutation
    permuted_data = permute(noised_data)

    # Apply scaling
    scaled_data = scale_data(permuted_data)

    # Apply warping
    augmented_data = warp_data(scaled_data)

    return augmented_data
