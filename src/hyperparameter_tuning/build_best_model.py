# build_best_model.py
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
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from utils.constants import (EPOCH, BATCH_SIZE, PATIENCE, MIN_DELTA)

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train, X_val, y_val, trial, device):
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Early stopping parameters
    patience = PATIENCE
    min_delta = MIN_DELTA
    early_stopping_counter = 0
    best_val_loss = float('inf')

    epochs = EPOCH
    batch_size = BATCH_SIZE

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Iterate over mini-batches
        for i in range(0, len(X_train), batch_size):
            # Get mini-batch
            X_batch = torch.Tensor(X_train[i:i + batch_size]).to(device)
            y_batch = torch.Tensor(y_train[i:i + batch_size]).to(device)

            # Forward pass
            model.train()
            optimizer.zero_grad()
            output = model(X_batch)

            # Compute loss
            loss = criterion(output, y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Evaluate on training set
        model.eval()
        with torch.no_grad():
            train_output = model(torch.Tensor(X_train).to(device))
            train_loss = criterion(train_output, torch.Tensor(y_train).to(device))
            train_losses.append(train_loss.item())

        # Evaluate on validation set
        with torch.no_grad():
            val_output = model(torch.Tensor(X_val).to(device))
            val_loss = criterion(val_output, torch.Tensor(y_val).to(device))
            val_losses.append(val_loss.item())

        logger.info(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Early stopping logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break
    history = {'train_loss': train_losses, 'val_loss': val_losses}

    return val_loss.item(), history
