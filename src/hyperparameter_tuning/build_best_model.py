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
from torch.utils.data import TensorDataset, DataLoader

from utils.constants import EPOCHS

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train, X_val, y_val, device):
    min_delta = model.min_delta
    patience = model.patience
    batch_size = model.batch_size
    optimizer = getattr(torch.optim, model.optimizer_name)(model.parameters(), lr=model.lr,
                                                           weight_decay=model.l2_regularizer)
    criterion = torch.nn.MSELoss()

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # Manually calculate L1 regularization and add it to the loss
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss += model.l1_regularizer * l1_penalty

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch_val, y_batch_val in val_loader:
                X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                val_output = model(X_batch_val)
                val_loss = criterion(val_output, y_batch_val)
                total_val_loss += val_loss.item() * X_batch_val.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)

        # Logging for each epoch
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    history = {'train_loss': train_losses, 'val_loss': val_losses}
    return best_val_loss, history

