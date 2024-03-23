# custom_models.py
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
import torch.nn as nn
import torch.nn.functional as F

'''-----------------------------ATTENTION Layers-------------------------------'''


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


class AttentionInTheMiddle(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionInTheMiddle, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(encoder_outputs))
        attention_scores = torch.matmul(energy, self.v)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(encoder_outputs * attention_weights, dim=1)
        return context_vector, attention_weights


class AttentionCNNLSTMGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionCNNLSTMGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs):
        # Correctly handling the expected dimensions for attention_scores calculation
        batch_size, seq_len, hidden_dim = encoder_outputs.size()
        energy = torch.tanh(self.attn(encoder_outputs))
        # Reshaping v to perform batch-wise dot product
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy.transpose(1, 2)).squeeze(1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(encoder_outputs * attention_weights, dim=1)
        return context_vector


class DeepAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DeepAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs):
        energy = torch.tanh(self.attn(encoder_outputs))
        attention_scores = torch.matmul(energy, self.v)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(encoder_outputs * attention_weights, dim=1)
        return context_vector, attention_weights


'''-----------------------------Simple models-------------------------------'''


class BuildLSTMModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildLSTMModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_lstm_layers = best_params['num_lstm_layers']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True))
            input_dim = lstm_units  # the input to the next layer

        self.fc = nn.Linear(in_features=lstm_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Average pooling across the sequence dimension
        x = torch.mean(x, dim=1)

        out = self.fc(x)
        return self.activation(out)


class BuildGRUModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildGRUModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
            gru_units = trial.suggest_int('gru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_gru_layers = best_params['num_gru_layers']
            gru_units = best_params['gru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(nn.GRU(input_size=input_dim, hidden_size=gru_units, batch_first=True))
            input_dim = gru_units  # the input to the next layer

        self.fc = nn.Linear(in_features=gru_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        # Average pooling across the sequence dimension
        x = torch.mean(x, dim=1)

        out = self.fc(x)
        return self.activation(out)


class BuildCNNModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(num_features=filters),
                getattr(nn, activation)(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension for the next layer

        self.cnn_layers = nn.Sequential(*cnn_layers)

        # ensure the output size matches for the fully connected layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(in_features=filters, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len)
        x = self.cnn_layers(x)

        # Apply adaptive pooling to make sure the output from CNN fits the fully connected layer
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)
        out = self.fc(x)  # Output shape: (batch_size, output_dim)
        return out


'''-----------------------------Simple Bi models-------------------------------'''


class BuildBiLSTMModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildBiLSTMModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_bilstm_layers = trial.suggest_int('num_bilstm_layers', 1, 3)
            bilstm_units = trial.suggest_int('bilstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_bilstm_layers = best_params['num_bilstm_layers']
            bilstm_units = best_params['bilstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.bilstm_layers = nn.ModuleList()
        for _ in range(num_bilstm_layers):
            self.bilstm_layers.append(
                nn.LSTM(input_size=input_dim, hidden_size=bilstm_units, batch_first=True, bidirectional=True))
            input_dim = bilstm_units * 2  # the input to the next layer

        self.fc = nn.Linear(in_features=bilstm_units * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for bilstm in self.bilstm_layers:
            x, _ = bilstm(x)
            x = self.dropout(x)

        # Average pooling across the sequence dimension
        x = torch.mean(x, dim=1)

        out = self.fc(x)
        return self.activation(out)


class BuildBiGRUModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildBiGRUModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_bigru_layers = trial.suggest_int('num_bigru_layers', 1, 3)
            bigru_units = trial.suggest_int('bigru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_bigru_layers = best_params['num_bigru_layers']
            bigru_units = best_params['bigru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.bigru_layers = nn.ModuleList()
        for _ in range(num_bigru_layers):
            self.bigru_layers.append(
                nn.GRU(input_size=input_dim, hidden_size=bigru_units, batch_first=True, bidirectional=True))
            input_dim = bigru_units * 2  # the input to the next layer

        self.fc = nn.Linear(in_features=bigru_units * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for bigru in self.bigru_layers:
            x, _ = bigru(x)
            x = self.dropout(x)

        # Average pooling across the sequence dimension
        x = torch.mean(x, dim=1)

        out = self.fc(x)
        return self.activation(out)


'''-----------------------------Simple + Attention models-------------------------------'''


class BuildLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildLSTMAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_lstm_layers = best_params['num_lstm_layers']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True))
            input_dim = lstm_units  # the input to the next layer

        self.attention = Attention(lstm_units)
        self.fc = nn.Linear(in_features=lstm_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)

        out = self.fc(context_vector)
        return self.activation(out)


class BuildGRUAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildGRUAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
            gru_units = trial.suggest_int('gru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_gru_layers = best_params['num_gru_layers']
            gru_units = best_params['gru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(nn.GRU(input_size=input_dim, hidden_size=gru_units, batch_first=True))
            input_dim = gru_units  # the input to the next layer

        self.attention = Attention(gru_units)
        self.fc = nn.Linear(in_features=gru_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)
        out = self.fc(context_vector)
        return self.activation(out)


class BuildCNNAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(num_features=filters),
                getattr(nn, activation)(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        # Adjusting for CNN output
        self.attention = Attention(filters)
        self.fc = nn.Linear(in_features=filters, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Adjusting for CNN input
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Adjusting for Attention input

        # Apply attention directly on CNN outputs
        context_vector = self.attention(x)
        context_vector = self.dropout(context_vector)

        out = self.fc(context_vector)
        return out


'''-----------------------------Bi + Attention models-------------------------------'''


class BuildBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildBiLSTMAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_bilstm_layers = trial.suggest_int('num_bilstm_layers', 1, 3)
            bilstm_units = trial.suggest_int('bilstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_bilstm_layers = best_params['num_bilstm_layers']
            bilstm_units = best_params['bilstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.bilstm_layers = nn.ModuleList()
        for _ in range(num_bilstm_layers):
            self.bilstm_layers.append(
                nn.LSTM(input_size=input_dim, hidden_size=bilstm_units, batch_first=True, bidirectional=True))
            input_dim = bilstm_units * 2  # the input to the next layer

        self.attention = Attention(bilstm_units * 2)
        self.fc = nn.Linear(in_features=bilstm_units * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for bilstm in self.bilstm_layers:
            x, _ = bilstm(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)

        out = self.fc(context_vector)
        return self.activation(out)


class BuildBiGRUAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildBiGRUAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            num_bigru_layers = trial.suggest_int('num_bigru_layers', 1, 3)
            bigru_units = trial.suggest_int('bigru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            num_bigru_layers = best_params['num_bigru_layers']
            bigru_units = best_params['bigru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        self.bigru_layers = nn.ModuleList()
        for _ in range(num_bigru_layers):
            self.bigru_layers.append(
                nn.GRU(input_size=input_dim, hidden_size=bigru_units, batch_first=True, bidirectional=True))
            input_dim = bigru_units * 2  # the input to the next layer

        self.attention = Attention(bigru_units * 2)
        self.fc = nn.Linear(in_features=bigru_units * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        for bigru in self.bigru_layers:
            x, _ = bigru(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)
        out = self.fc(context_vector)
        return self.activation(out)


'''-----------------------------Hybrid models-------------------------------'''


class BuildCNNLSTMModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNLSTMModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_lstm_layers = best_params['num_lstm_layers']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True))
            input_dim = lstm_units

        self.fc = nn.Linear(in_features=lstm_units, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Permute x to (batch_size, input_dim, seq_len) for CNN
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        # Permute x back to (batch_size, seq_len, new_dim) for LSTM
        x = x.permute(0, 2, 1)

        # Process through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Assuming we use the output of the last LSTM layer's last time step
        x = x[:, -1, :]

        x = self.fc(x)
        return self.activation(x)


class BuildCNNGRUModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNGRUModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
            gru_units = trial.suggest_int('gru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_gru_layers = best_params['num_gru_layers']
            gru_units = best_params['gru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(nn.GRU(input_size=input_dim, hidden_size=gru_units, batch_first=True))
            input_dim = gru_units

        self.fc = nn.Linear(in_features=gru_units, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Permute x to (batch_size, input_dim, seq_len) for CNN
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        # Permute x back to (batch_size, seq_len, new_dim)
        x = x.permute(0, 2, 1)

        # Process through LSTM layers
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        x = x[:, -1, :]

        x = self.fc(x)
        return self.activation(x)


class BuildCNNBiLSTMModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNBiLSTMModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_bilstm_layers = trial.suggest_int('num_bilstm_layers', 1, 3)
            bilstm_units = trial.suggest_int('bilstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_bilstm_layers = best_params['num_bilstm_layers']
            bilstm_units = best_params['bilstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.bilstm_layers = nn.ModuleList()
        for _ in range(num_bilstm_layers):
            self.bilstm_layers.append(
                nn.LSTM(input_size=input_dim, hidden_size=bilstm_units, batch_first=True, bidirectional=True))
            input_dim = bilstm_units * 2

        self.fc = nn.Linear(in_features=bilstm_units * 2, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Permute x to (batch_size, input_dim, seq_len) for CNN
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        # Permute x back to (batch_size, seq_len, new_dim) for LSTM
        x = x.permute(0, 2, 1)

        # Process through LSTM layers
        for bilstm in self.bilstm_layers:
            x, (hidden, cell) = bilstm(x)
            x = self.dropout(x)

        # Assuming we use the output of the last LSTM layer's last time step
        x = x[:, -1, :]

        x = self.fc(x)
        return self.activation(x)


class BuildCNNBiGRUModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNBiGRUModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_bigru_layers = trial.suggest_int('num_bigru_layers', 1, 3)
            bigru_units = trial.suggest_int('bigru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_bigru_layers = best_params['num_bigru_layers']
            bigru_units = best_params['bigru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.bigru_layers = nn.ModuleList()
        for _ in range(num_bigru_layers):
            self.bigru_layers.append(
                nn.GRU(input_size=input_dim, hidden_size=bigru_units, batch_first=True, bidirectional=True))
            input_dim = bigru_units * 2

        self.fc = nn.Linear(in_features=bigru_units * 2, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Permute x to (batch_size, input_dim, seq_len) for CNN
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        # Permute x back to (batch_size, seq_len, new_dim) for LSTM
        x = x.permute(0, 2, 1)

        # Process through LSTM layers
        for bigru in self.bigru_layers:
            x, (hidden, cell) = bigru(x)
            x = self.dropout(x)

        # Assuming we use the output of the last LSTM layer's last time step
        x = x[:, -1, :]

        x = self.fc(x)
        return self.activation(x)


'''-----------------------------Hybrid + Attention models-------------------------------'''


class BuildCNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNLSTMAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_lstm_layers = best_params['num_lstm_layers']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True))
            input_dim = lstm_units

        self.attention = Attention(lstm_units)
        self.fc = nn.Linear(in_features=lstm_units, out_features=output_dim)

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


class BuildCNNGRUAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNGRUAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
            gru_units = trial.suggest_int('gru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_gru_layers = best_params['num_gru_layers']
            gru_units = best_params['gru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(nn.GRU(input_size=input_dim, hidden_size=gru_units, batch_first=True))
            input_dim = gru_units

        self.attention = Attention(gru_units)
        self.fc = nn.Linear(in_features=gru_units, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim)

        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)

        out = self.fc(context_vector)  # Output shape: (batch_size, output_dim)
        return self.activation(out)


class BuildCNNBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNBiLSTMAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_bilstm_layers = trial.suggest_int('num_bilstm_layers', 1, 3)
            bilstm_units = trial.suggest_int('bilstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_bilstm_layers = best_params['num_bilstm_layers']
            bilstm_units = best_params['bilstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.bilstm_layers = nn.ModuleList()
        for _ in range(num_bilstm_layers):
            self.bilstm_layers.append(
                nn.LSTM(input_size=input_dim, hidden_size=bilstm_units, batch_first=True, bidirectional=True))
            input_dim = bilstm_units * 2

        self.attention = Attention(bilstm_units * 2)
        self.fc = nn.Linear(in_features=bilstm_units * 2, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim)

        for bilstm in self.bilstm_layers:
            x, _ = bilstm(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)

        out = self.fc(context_vector)  # Output shape: (batch_size, output_dim)
        return self.activation(out)


class BuildCNNBiGRUAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNBiGRUAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_bigru_layers = trial.suggest_int('num_bigru_layers', 1, 3)
            bigru_units = trial.suggest_int('bigru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_bigru_layers = best_params['num_bigru_layers']
            bigru_units = best_params['bigru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters  # next input dimension

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.bigru_layers = nn.ModuleList()
        for _ in range(num_bigru_layers):
            self.bigru_layers.append(
                nn.GRU(input_size=input_dim, hidden_size=bigru_units, batch_first=True, bidirectional=True))
            input_dim = bigru_units * 2

        self.attention = Attention(bigru_units * 2)
        self.fc = nn.Linear(in_features=bigru_units * 2, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim)

        for bigru in self.bigru_layers:
            x, _ = bigru(x)
            x = self.dropout(x)

        # Apply attention
        context_vector = self.attention(x)

        out = self.fc(context_vector)  # Output shape: (batch_size, output_dim)
        return self.activation(out)


'''-----------------------------Deep Hybrid + Attention models-------------------------------'''


class BuildCNNAttentionLSTMModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionLSTMModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_lstm_layers = best_params['num_lstm_layers']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.attention = AttentionInTheMiddle(input_dim)

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True))
            input_dim = lstm_units

        self.fc = nn.Linear(in_features=lstm_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len) for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim) for attention

        # Apply attention
        context_vector, attention_weights = self.attention(x)
        seq_len = x.size(1)  # Assuming x is the input to attention with shape (batch_size, seq_len, hidden_dim)
        context_vector_expanded = context_vector.unsqueeze(1).repeat(1, seq_len, 1)

        # Now, context_vector_expanded can be used as input to LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(context_vector_expanded)
            context_vector_expanded = self.dropout(x)

        # Final layers
        out = self.fc(x[:, -1, :])
        return self.activation(out)


class BuildCNNAttentionGRUModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionGRUModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
            gru_units = trial.suggest_int('gru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_gru_layers = best_params['num_gru_layers']
            gru_units = best_params['gru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.attention = AttentionInTheMiddle(input_dim)

        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(nn.GRU(input_size=input_dim, hidden_size=gru_units, batch_first=True))
            input_dim = gru_units

        self.fc = nn.Linear(in_features=gru_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len) for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim) for attention

        # Apply attention
        context_vector, attention_weights = self.attention(x)
        seq_len = x.size(1)  # Assuming x is the input to attention with shape (batch_size, seq_len, hidden_dim)
        context_vector_expanded = context_vector.unsqueeze(1).repeat(1, seq_len, 1)

        # Now, context_vector_expanded can be used as input to LSTM layers
        for gru in self.gru_layers:
            x, _ = gru(context_vector_expanded)
            context_vector_expanded = self.dropout(x)

        # Final layers
        out = self.fc(x[:, -1, :])
        return self.activation(out)


class BuildCNNAttentionBiLSTMModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionBiLSTMModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_bilstm_layers = trial.suggest_int('num_bilstm_layers', 1, 3)
            bilstm_units = trial.suggest_int('bilstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_bilstm_layers = best_params['num_bilstm_layers']
            bilstm_units = best_params['bilstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.attention = AttentionInTheMiddle(input_dim)

        self.bilstm_layers = nn.ModuleList()
        for _ in range(num_bilstm_layers):
            self.bilstm_layers.append(
                nn.LSTM(input_size=input_dim, hidden_size=bilstm_units, batch_first=True, bidirectional=True))
            input_dim = bilstm_units * 2

        self.fc = nn.Linear(in_features=bilstm_units * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len) for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim) for attention

        # Apply attention
        context_vector, attention_weights = self.attention(x)
        seq_len = x.size(1)  # Assuming x is the input to attention with shape (batch_size, seq_len, hidden_dim)
        context_vector_expanded = context_vector.unsqueeze(1).repeat(1, seq_len, 1)

        # Now, context_vector_expanded can be used as input to LSTM layers
        for bilstm in self.bilstm_layers:
            x, _ = bilstm(context_vector_expanded)
            context_vector_expanded = self.dropout(x)

        # Final layers
        out = self.fc(x[:, -1, :])
        return self.activation(out)


class BuildCNNAttentionBiGRUModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionBiGRUModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_bigru_layers = trial.suggest_int('num_bigru_layers', 1, 3)
            bigru_units = trial.suggest_int('bigru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_bigru_layers = best_params['num_bigru_layers']
            bigru_units = best_params['bigru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.attention = AttentionInTheMiddle(input_dim)

        self.bigru_layers = nn.ModuleList()
        for _ in range(num_bigru_layers):
            self.bigru_layers.append(
                nn.GRU(input_size=input_dim, hidden_size=bigru_units, batch_first=True, bidirectional=True))
            input_dim = bigru_units * 2

        self.fc = nn.Linear(in_features=bigru_units * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len) for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, hidden_dim) for attention

        # Apply attention
        context_vector, attention_weights = self.attention(x)
        seq_len = x.size(1)  # input to attention with shape (batch_size, seq_len, hidden_dim)
        context_vector_expanded = context_vector.unsqueeze(1).repeat(1, seq_len, 1)

        # Now, context_vector_expanded can be used as input to LSTM layers
        for bigru in self.bigru_layers:
            x, _ = bigru(context_vector_expanded)
            context_vector_expanded = self.dropout(x)

        # Final layers
        out = self.fc(x[:, -1, :])
        return self.activation(out)


'''-----------------------------Deep More Hybrid + Attention models-------------------------------'''


class BuildCNNAttentionLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionLSTMAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_lstm_layers = best_params['num_lstm_layers']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(1, num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.attention_after_cnn = AttentionCNNLSTMGRU(filters)  # Ensure filters is correct for attention input

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True))
            input_dim = lstm_units

        self.attention_after_lstm = Attention(lstm_units)

        self.fc = nn.Linear(in_features=lstm_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # CNN layers
        x = x.permute(0, 2, 1)  # Change for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Revert for attention

        # First attention mechanism
        context_vector = self.attention_after_cnn(x)
        x = context_vector.unsqueeze(1).repeat(1, x.size(1), 1)  # Prepare for LSTM

        # LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Second attention mechanism
        context_vector_after_lstm = self.attention_after_lstm(x)
        x = context_vector_after_lstm

        # Final layers
        out = self.fc(x)
        return self.activation(out)


class BuildCNNAttentionGRUAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None, X_train=None, output_dim=None):
        super(BuildCNNAttentionGRUAttentionModel, self).__init__()

        if trial:
            self.lr = trial.suggest_categorical('lr', [1e-4, 1e-3])
            self.optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            self.l1_regularizer = trial.suggest_categorical('l1_regularizer', [1e-5, 1e-4, 1e-3])
            self.l2_regularizer = trial.suggest_categorical('l2_regularizer', [1e-5, 1e-4, 1e-3])
            self.min_delta = trial.suggest_categorical('min_delta', [1e-8, 1e-7])
            self.patience = trial.suggest_int('patience', 30, 50, step=5)
            self.batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
            filters = trial.suggest_categorical('filters', [32, 64, 96])
            input_dim = X_train.shape[2] if trial else input_dim
            if output_dim < 2:
                kernel_size = 1
                pool_size = 1
            else:
                # Generate kernel_size options based on output_dim
                kernel_size_options = [2, 3][:output_dim]
                kernel_size = trial.suggest_categorical('kernel_size', kernel_size_options)
                pool_size = trial.suggest_int('pool_size', 1, min(1, kernel_size))
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
            gru_units = trial.suggest_int('gru_units', 50, 200, step=50)
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
        else:
            self.lr = best_params['lr']
            self.optimizer_name = best_params['optimizer']
            self.l1_regularizer = best_params['l1_regularizer']
            self.l2_regularizer = best_params['l2_regularizer']
            self.min_delta = best_params['min_delta']
            self.patience = best_params['patience']
            self.batch_size = best_params['batch_size']
            filters = best_params['filters']
            output_dim = output_dim or X_train.shape[2]
            if 'kernel_size' in best_params:
                kernel_size = min(best_params['kernel_size'], output_dim)
            else:
                kernel_size = 1 if output_dim <= 1 else 2

            if 'pool_size' in best_params:
                pool_size = min(best_params['pool_size'], kernel_size, output_dim)
            else:
                pool_size = 1 if kernel_size == 1 else min(2, output_dim)
            num_cnn_layers = best_params['num_cnn_layers']
            num_gru_layers = best_params['num_gru_layers']
            gru_units = best_params['gru_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
        input_dim = input_dim if input_dim is not None else X_train.shape[2]

        cnn_layers = []
        for _ in range(1, num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=filters, kernel_size=kernel_size),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            ])
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.attention_after_cnn = AttentionCNNLSTMGRU(filters)  # Ensure filters is correct for attention input

        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(nn.LSTM(input_size=input_dim, hidden_size=gru_units, batch_first=True))
            input_dim = gru_units

        self.attention_after_gru = Attention(gru_units)

        self.fc = nn.Linear(in_features=gru_units, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        # CNN layers
        x = x.permute(0, 2, 1)  # Change for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Revert for attention

        # First attention mechanism
        context_vector = self.attention_after_cnn(x)
        x = context_vector.unsqueeze(1).repeat(1, x.size(1), 1)  # Prepare for LSTM

        # LSTM layers
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        # Second attention mechanism
        context_vector_after_gru = self.attention_after_gru(x)
        x = context_vector_after_gru

        # Final layers
        out = self.fc(x)
        return self.activation(out)
