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


def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def permute(data, permute_size):
    permuted_data = np.copy(data)
    for i in range(data.shape[0]):
        start = np.random.randint(0, data.shape[1] - permute_size)
        permuted_section = permuted_data[i, start:start + permute_size]
        np.random.shuffle(permuted_section)
        permuted_data[i, start:start + permute_size] = permuted_section
    return permuted_data


def scale_data(data, scaling_factor):
    return data * scaling_factor


def warp_data(data, warp_strength):
    warped_data = np.copy(data)
    # Apply some warping function based on warp_strength
    # This is a placeholder for the actual warping logic
    return warped_data


def robust_data_augmentation(dataset, noise_level=0.01, permute_size=5, scaling_factor=1.1, warp_strength=0.1):
    # Apply noise
    noised_data = add_noise(dataset, noise_level=noise_level)

    # Apply permutation
    permuted_data = permute(noised_data, permute_size=permute_size)

    # Apply scaling
    scaled_data = scale_data(permuted_data, scaling_factor=scaling_factor)

    # Apply warping
    augmented_data = warp_data(scaled_data, warp_strength=warp_strength)

    return augmented_data
