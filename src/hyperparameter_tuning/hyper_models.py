# hyper_models.py

import torch
import torch.nn as nn


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

class BuildCNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, num_cnn_layers=None, num_lstm_layers=None, filters=None, kernel_size=None,
                 lstm_units=None, dropout=None, activation=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNLSTMAttentionModel, self).__init__()

        if trial is not None:
            default_input_dim = X_train.shape[2]
            default_num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            default_num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            default_filters = trial.suggest_categorical('filters', [32, 64, 96, 128])
            default_kernel_size = trial.suggest_categorical('kernel_size', [1, 2, 3])
            default_lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            default_dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
            default_activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh', 'ELU'])

        # Use default values if not provided
        input_dim = input_dim or default_input_dim
        num_cnn_layers = num_cnn_layers or default_num_cnn_layers
        num_lstm_layers = num_lstm_layers or default_num_lstm_layers
        filters = filters or default_filters
        kernel_size = kernel_size or default_kernel_size
        lstm_units = lstm_units or default_lstm_units
        dropout = dropout or default_dropout
        activation = activation or default_activation

        # Rest of your initialization code...


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