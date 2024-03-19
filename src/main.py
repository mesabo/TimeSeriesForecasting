import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import optuna
import json


def load_and_create_dataset():
    # Generate synthetic dataset with 1000 rows and 4 columns
    num_rows = 100
    num_columns = 4
    dataset = np.random.rand(num_rows, num_columns)
    # Convert to DataFrame
    dataset = pd.DataFrame(dataset, columns=['Feature1', 'Feature2', 'Feature3', 'Target'])
    return dataset


def normalize_and_split_dataset(dataset, look_back, forecast_days):
    # Separate features and target variable
    features = dataset.drop(columns=['Target']).values
    target = dataset['Target'].values.reshape(-1, 1)

    # Normalize features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    # Normalize target variable
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    # Combine scaled features and target variable
    scaled_dataset = np.concatenate((scaled_features, scaled_target), axis=1)

    # Split into input and output
    X, y = [], []
    for i in range(len(scaled_dataset) - look_back - forecast_days + 1):
        X.append(scaled_dataset[i: (i + look_back), :-1])
        y.append(scaled_dataset[i + look_back: i + look_back + forecast_days, -1])

    X, y = np.array(X), np.array(y)

    # Split into train and test sets
    train_size = int(len(X) * 0.2)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler_features, scaler_target


class PyTorchModel(nn.Module):
    def __init__(self, input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,
                 activation, output_dim):
        super(PyTorchModel, self).__init__()
        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.append(nn.Conv1d(input_dim, filters, kernel_size))
            cnn_layers.append(nn.ReLU())
            input_dim = filters
        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_dim, lstm_units, batch_first=True))
            input_dim = lstm_units

        self.fc = nn.Linear(lstm_units, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim)

        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        out = self.fc(x[:, -1, :])  # Take the last time step's output
        return self.activation(out)


def train_model(model, X_train, y_train, X_val, y_val, trial, device):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer = getattr(optim, trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop']))(model.parameters(),
                                                                                                   lr=lr)
    criterion = nn.MSELoss()

    # Early stopping parameters
    patience = 5
    min_delta = 0.001
    early_stopping_counter = 0
    best_val_loss = float('inf')

    epochs = 10  # Maximum number of epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

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


def objective(trial):
    look_back = 30
    forecast_days = 1

    # Load and split dataset
    dataset = load_and_create_dataset()
    X_train, X_test, y_train, y_test, _, _ = normalize_and_split_dataset(dataset, look_back, forecast_days)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(X_train.shape)
    # Create PyTorch model
    input_dim = X_train.shape[2]
    num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    filters = trial.suggest_categorical('filters', [32, 64, 96, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [1, 2, 3])
    lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh', 'ELU'])
    output_dim = forecast_days
    model = PyTorchModel(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,
                         activation, output_dim)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Move data to GPU
    X_train, y_train = torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device)
    X_val, y_val = torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device)

    # Train the model
    val_loss = train_model(model, X_train, y_train, X_val, y_val, trial, device)

    return val_loss

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(study.best_trial.params)

    # Extract best hyperparameters
    best_params = study.best_trial.params

    # Load and split dataset
    dataset = load_and_create_dataset()
    X_train, X_test, y_train, y_test, _, _ = normalize_and_split_dataset(dataset, look_back=7, forecast_days=2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create PyTorch model with best hyperparameters
    input_dim = X_train.shape[2]
    model = PyTorchModel(input_dim, best_params['num_cnn_layers'], best_params['num_lstm_layers'],
                         best_params['filters'], best_params['kernel_size'], best_params['lstm_units'],
                         best_params['dropout'], best_params['activation'], output_dim=2)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Move data to GPU
    X_test, y_test = torch.Tensor(X_test).to(device), torch.Tensor(y_test).to(device)

    # Train the best model on full training set
    val_loss = train_model(model, torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device),
                           torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device),
                           optuna.trial.FixedTrial(best_params), device)

    # Calculate RMSE on test set
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test)
        rmse = torch.sqrt(test_loss).item()

    print("RMSE of the best model:", rmse)

    output_file_path = f"./output/sample.json"
    with open(output_file_path, "w") as output_file:
        json.dump(study.best_trial.params, output_file)

    print("Output saved to:", output_file_path)


if __name__ == "__main__":
    main()
