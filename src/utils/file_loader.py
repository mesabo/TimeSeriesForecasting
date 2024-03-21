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
import json


def read_features(features_path, key_name):
    with open(features_path, "r") as file:
        data_features = json.load(file)

    if key_name in data_features:
        features = data_features[key_name]
        return features
    else:
        raise KeyError(f"Key '{key_name}' not found in the JSON file.")


def read_best_params(features_path, key_name):
    with open(features_path, "r") as file:
        data_features = json.load(file)

    if key_name in data_features:
        features = data_features[key_name]
        return features
    else:
        raise KeyError(f"MODELS '{key_name}' not found in the JSON file.")
