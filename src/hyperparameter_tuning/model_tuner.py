# model_tuner.py

import torch
from hyperparameter_tuning.model_training import train_model
from hyperparameter_tuning.hyper_models import BuildCNNLSTMAttentionModel
from utils.constants import (
    LSTM_MODEL, GRU_MODEL, CNN_MODEL, BiLSTM_MODEL, BiGRU_MODEL,
    LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL, CNN_ATTENTION_MODEL,
    BiLSTM_ATTENTION_MODEL, BiGRU_ATTENTION_MODEL,
    CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
    CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_ATTENTION_MODEL, CNN_ATTENTION_GRU_ATTENTION_MODEL,
    CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_MODEL, CNN_ATTENTION_GRU_MODEL,
    CNN_ATTENTION_BiLSTM_MODEL, CNN_ATTENTION_BiGRU_MODEL,
    SAVING_MODEL_DIR, SAVING_METRIC_DIR, SAVING_LOSS_DIR, BASE_PATH,
    SAVING_PREDICTION_DIR, SAVING_METRICS_PATH, SAVING_LOSSES_PATH, SEEDER,
    HYPERBAND_PATH, DATASET_FEATURES_PATH, ELECTRICITY_DATASET_PATH,
    ELECTRICITY
)



class ModelTuner:
    def __init__(self, X_train, y_train, X_val, y_val, output_dim, model_type):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.output_dim = output_dim
        self.model_type = model_type

    def objective(self, trial):
        model = self.build_model(trial)

        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Move data to GPU
        X_train, y_train = torch.Tensor(self.X_train).to(device), torch.Tensor(self.y_train).to(device)
        X_val, y_val = torch.Tensor(self.X_val).to(device), torch.Tensor(self.y_val).to(device)

        # Train the model
        val_loss = train_model(model, X_train, y_train, X_val, y_val, trial, device)

        return val_loss

    def build_model(self, trial):
        if self.model_type == CNN_LSTM_ATTENTION_MODEL:
            return BuildCNNLSTMAttentionModel(trial=trial, X_train=self.X_train, output_dim=self.output_dim)
        elif self.model_type == GRU_MODEL:
            pass
            # return build_gru_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_MODEL:
            pass
            # return build_cnn_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)

        # -----------------------------Simple Bi models-------------------------------
        elif self.model_type == BiLSTM_MODEL:
            pass
            # return build_bilstm_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == BiGRU_MODEL:
            pass
            # return build_bigru_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)

        # -----------------------------Simple + Attention models-------------------------------
        elif self.model_type == LSTM_ATTENTION_MODEL:
            pass
            # return build_lstm_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == GRU_ATTENTION_MODEL:
            pass
            # return build_gru_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_MODEL:
            pass
            # return build_cnn_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        # -----------------------------Bi + Attention models-------------------------------
        elif self.model_type == BiLSTM_ATTENTION_MODEL:
            pass
            # return build_bilstm_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == BiGRU_ATTENTION_MODEL:
            pass
            # return build_bigru_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        # -----------------------------Hybrid models-------------------------------
        elif self.model_type == CNN_LSTM_MODEL:
            pass
            # return build_cnn_lstm_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_GRU_MODEL:
            pass
            # return build_cnn_gru_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_BiLSTM_MODEL:
            pass
            # return build_cnn_bilstm_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_BiGRU_MODEL:
            pass
            # return build_cnn_bigru_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        # -----------------------------Hybrid + Attention models-------------------------------
        elif self.model_type == CNN_LSTM_ATTENTION_MODEL:
            pass
            # return build_cnn_lstm_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_GRU_ATTENTION_MODEL:
            pass
            # return build_cnn_gru_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_BiLSTM_ATTENTION_MODEL:
            pass
            # return build_cnn_bilstm_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_BiGRU_ATTENTION_MODEL:
            pass
            # return build_cnn_bigru_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        # -----------------------------Deep Hybrid + Attention models-------------------------------
        elif self.model_type == CNN_ATTENTION_LSTM_MODEL:
            pass
            # return build_cnn_attention_lstm_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_GRU_MODEL:
            pass
            # return build_cnn_attention_gru_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiLSTM_MODEL:
            pass
            # return build_cnn_attention_bilstm_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiGRU_MODEL:
            pass
            # return build_cnn_attention_bigru_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        # -----------------------------Deep More Hybrid + Attention models-------------------------------
        elif self.model_type == CNN_ATTENTION_LSTM_ATTENTION_MODEL:
            pass
            # return build_cnn_attention_lstm_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_GRU_ATTENTION_MODEL:
            pass
            # return build_cnn_attention_gru_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiLSTM_ATTENTION_MODEL:
            pass
            # return build_cnn_attention_bilstm_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiGRU_ATTENTION_MODEL:
            pass
            # return build_cnn_attention_bigru_attention_model(input_dim, num_cnn_layers, num_lstm_layers, filters, kernel_size, lstm_units, dropout,activation, self.output_dim)
        else:
            raise ValueError(
                "Invalid model type. Please choose from the available models.")
