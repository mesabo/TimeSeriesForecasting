# model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.constants import EPOCH
def train_model(model, X_train, y_train, X_val, y_val, trial, device):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop']))(model.parameters(),
                                                                                                   lr=lr)
    criterion = nn.MSELoss()

    # Early stopping parameters
    patience = 5
    min_delta = 0.001
    early_stopping_counter = 0
    best_val_loss = float('inf')

    epochs = 2  # EPOCH
    batch_size = 64  # Choose an appropriate batch size

    for epoch in range(epochs):
        # Iterate over mini-batches
        for i in range(0, X_train.size(0), batch_size):
            # Get mini-batch
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            model.train()
            optimizer.zero_grad()
            output = model(X_batch)

            # Compute loss
            loss = criterion(output, y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Optionally, evaluate model on validation set after each epoch

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Early stopping logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    return val_loss.item()
