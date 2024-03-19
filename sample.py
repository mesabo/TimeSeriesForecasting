import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import json
from src.input_processing.data_processing import preprocess_and_split_dataset, preprocess_augment_and_split_dataset
from src.utils.constants import ELECTRICITY

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(encoder_outputs))
        attention_scores = torch.matmul(energy, self.v)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(encoder_outputs * attention_weights, dim=1)
        return context_vector

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

        self.attention = Attention(lstm_units)
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

        # Apply attention
        context_vector = self.attention(x)

        out = self.fc(context_vector)  # Output shape: (batch_size, output_dim)
        return self.activation(out)

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

    epochs = 2  # Maximum number of epochs
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
    forecast_days = 7

    # Load and split dataset
    X_train, X_val, y_train, y_val, _ = preprocess_augment_and_split_dataset(ELECTRICITY, 'D', look_back, forecast_days)

    # Create PyTorch model
    print(X_train.shape, y_train.shape)
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
    X_train, X_val, y_train, y_val, _ = preprocess_augment_and_split_dataset(ELECTRICITY, 'D', look_back=7, forecast_period=2)

    # Create PyTorch model with best hyperparameters
    input_dim = X_train.shape[2]
    model = PyTorchModel(input_dim, best_params['num_cnn_layers'], best_params['num_lstm_layers'],
                         best_params['filters'], best_params['kernel_size'], best_params['lstm_units'],
                         best_params['dropout'], best_params['activation'], output_dim=2)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Move data to GPU
    X_val, y_val = torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device)

    # Train the best model on full training set
    val_loss = train_model(model, torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device),
                           X_val, y_val,
                           optuna.trial.FixedTrial(best_params), device)

    # Calculate RMSE on test set
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_output = model(X_val)
        test_loss = criterion(test_output, y_val)
        rmse = torch.sqrt(test_loss).item()

    print("RMSE of the best model:", rmse)

    output_file_path = f"./output/sample.json"
    with open(output_file_path, "w") as output_file:
        json.dump(study.best_trial.params, output_file)

    print("Output saved to:", output_file_path)


if __name__ == "__main__":
    main()
