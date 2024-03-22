# model_tuner.py
# !/usr/bin/env python3
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

import torch

from hyperparameter_tuning.build_best_model import train_model
from models.model_selection import ModelSelection


class ModelTuner:
    def __init__(self, X_train, y_train, X_val, y_val, output_dim, model_type):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.output_dim = output_dim
        self.model_type = model_type
        self.model_selection = ModelSelection(X_train, output_dim, model_type)

    def objective(self, trial):
        # Build the model with the current trial's suggestions or best known parameters
        model = self.model_selection.select_model(trial, best_params=None)

        # Move model to the appropriate device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Convert data to tensors and move to the same device as the model
        X_train_tensor = torch.Tensor(self.X_train).to(device)
        y_train_tensor = torch.Tensor(self.y_train).to(device)
        X_val_tensor = torch.Tensor(self.X_val).to(device)
        y_val_tensor = torch.Tensor(self.y_val).to(device)

        # Train the model and get the validation loss
        val_loss, _ = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, device)

        # The goal is to minimize the validation loss
        return val_loss
