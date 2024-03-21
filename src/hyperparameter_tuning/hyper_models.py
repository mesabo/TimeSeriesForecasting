# hyper_models.py
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


'''-----------------------------Simple models-------------------------------'''

'''-----------------------------Simple Bi models-------------------------------'''

'''-----------------------------Simple + Attention models-------------------------------'''

'''-----------------------------Bi + Attention models-------------------------------'''

'''-----------------------------Hybrid models-------------------------------'''


class BuildCNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=None, best_params=None, trial=None,
                 X_train=None, output_dim=None):
        super(BuildCNNLSTMAttentionModel, self).__init__()

        if trial is not None:
            input_dim = X_train.shape[2]
            num_cnn_layers = trial.suggest_int('num_cnn_layers', 1, 3)
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            filters = trial.suggest_categorical('filters', [32, 64, 96, 128])
            kernel_size = trial.suggest_categorical('kernel_size', [1, 2, 3])
            pool_kernel_size = trial.suggest_categorical('pool_kernel_size', [1, 2, 3, 4])
            stride = trial.suggest_categorical('stride', [1, 2, 3])
            padding = trial.suggest_categorical('padding', ['same', 'valid'])
            dilation = trial.suggest_categorical('dilation', [1, 2])
            lstm_units = trial.suggest_int('lstm_units', 50, 200, step=50)
            dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
            activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
            l1_regularizer = trial.suggest_float('l1_regularizer', 1e-6, 1e-2, log=True)
            l2_regularizer = trial.suggest_float('l2_regularizer', 1e-6, 1e-2, log=True)

        else:
            # Use best_params if provided meaning trial is None
            input_dim = input_dim or X_train.shape[2]
            num_cnn_layers = best_params['num_cnn_layers']
            num_lstm_layers = best_params['num_lstm_layers']
            filters = best_params['filters']
            kernel_size = best_params['kernel_size']
            pool_kernel_size = best_params['pool_kernel_size']
            stride = best_params['stride']
            padding = best_params['padding']
            dilation = best_params['dilation']
            lstm_units = best_params['lstm_units']
            dropout = best_params['dropout']
            activation = best_params['activation']
            l1_regularizer = best_params['l1_regularizer']
            l2_regularizer = best_params['l2_regularizer']

        cnn_layers = []
        for _ in range(num_cnn_layers):
            if padding == 'same':
                # Calculate padding to ensure output has the same length as input
                padding_value = (pool_kernel_size - 1) // 2
            else:
                padding_value = 0
            cnn_layers.append(nn.Conv1d(input_dim, filters, min(kernel_size, input_dim), padding=padding_value))
            cnn_layers.append(nn.BatchNorm1d(filters))
            cnn_layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size or 2, stride=stride, padding=padding_value,
                                           dilation=dilation))
            input_dim = filters

        self.cnn_layers = nn.Sequential(*cnn_layers)

        self.lstm_layers = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_dim, lstm_units, batch_first=True))
            input_dim = lstm_units

        self.attention = Attention(lstm_units)
        self.fc = nn.Linear(lstm_units, output_dim)
        self.fc.weight_regularizer = l2_regularizer
        self.fc.bias_regularizer = l1_regularizer

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


'''-----------------------------Deep Hybrid + Attention models-------------------------------'''

'''-----------------------------Deep More Hybrid + Attention models-------------------------------'''
