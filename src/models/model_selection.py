#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/03/2024
🚀 Welcome to the Awesome Python Script 🚀

User: mesabo
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab

"""
from models.custom_models import (BuildCNNLSTMAttentionModel, BuildLSTMModel, BuildGRUModel, BuildCNNModel,
                                  BuildBiLSTMModel, BuildBiGRUModel, BuildLSTMAttentionModel, BuildGRUAttentionModel,
                                  BuildCNNAttentionModel, BuildBiLSTMAttentionModel, BuildBiGRUAttentionModel,
                                  BuildCNNLSTMModel, BuildCNNGRUModel, BuildCNNBiLSTMModel, BuildCNNBiGRUModel,
                                  BuildCNNGRUAttentionModel, BuildCNNBiLSTMAttentionModel, BuildCNNBiGRUAttentionModel,
                                  BuildCNNAttentionLSTMModel, BuildCNNAttentionGRUModel, BuildCNNAttentionBiLSTMModel,
                                  BuildCNNAttentionBiGRUModel)
from utils.constants import (
    LSTM_MODEL, GRU_MODEL, CNN_MODEL, BiLSTM_MODEL, BiGRU_MODEL,
    LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL, CNN_ATTENTION_MODEL,
    BiLSTM_ATTENTION_MODEL, BiGRU_ATTENTION_MODEL,
    CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
    CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_MODEL, CNN_ATTENTION_GRU_MODEL,
    CNN_ATTENTION_BiLSTM_MODEL, CNN_ATTENTION_BiGRU_MODEL
)


class ModelSelection:
    def __init__(self, X_train, output_dim, model_type):
        self.X_train = X_train
        self.output_dim = output_dim
        self.model_type = model_type

    def select_model(self, trial=None, best_params=None):
        if self.model_type == LSTM_MODEL:
            return BuildLSTMModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                  output_dim=self.output_dim)
        elif self.model_type == GRU_MODEL:
            return BuildGRUModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                 output_dim=self.output_dim)
        elif self.model_type == CNN_MODEL:
            return BuildCNNModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                 output_dim=self.output_dim)

        # -----------------------------Simple Bi models-------------------------------
        elif self.model_type == BiLSTM_MODEL:
            return BuildBiLSTMModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                    output_dim=self.output_dim)
        elif self.model_type == BiGRU_MODEL:
            return BuildBiGRUModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                   output_dim=self.output_dim)

        # -----------------------------Simple + Attention models-------------------------------
        elif self.model_type == LSTM_ATTENTION_MODEL:
            return BuildLSTMAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                           output_dim=self.output_dim)
        elif self.model_type == GRU_ATTENTION_MODEL:
            return BuildGRUAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                          output_dim=self.output_dim)
        elif self.model_type == CNN_ATTENTION_MODEL:
            return BuildCNNAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                          output_dim=self.output_dim)
        # -----------------------------Bi + Attention models-------------------------------
        elif self.model_type == BiLSTM_ATTENTION_MODEL:
            return BuildBiLSTMAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                             output_dim=self.output_dim)
        elif self.model_type == BiGRU_ATTENTION_MODEL:
            return BuildBiGRUAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                            output_dim=self.output_dim)
        # -----------------------------Hybrid models-------------------------------
        elif self.model_type == CNN_LSTM_MODEL:
            return BuildCNNLSTMModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                     output_dim=self.output_dim)
        elif self.model_type == CNN_GRU_MODEL:
            return BuildCNNGRUModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                    output_dim=self.output_dim)
        elif self.model_type == CNN_BiLSTM_MODEL:
            return BuildCNNBiLSTMModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                       output_dim=self.output_dim)
        elif self.model_type == CNN_BiGRU_MODEL:
            return BuildCNNBiGRUModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                      output_dim=self.output_dim)
        # -----------------------------Hybrid + Attention models-------------------------------
        elif self.model_type == CNN_LSTM_ATTENTION_MODEL:
            return BuildCNNLSTMAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                              output_dim=self.output_dim)
        elif self.model_type == CNN_GRU_ATTENTION_MODEL:
            return BuildCNNGRUAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                             output_dim=self.output_dim)
        elif self.model_type == CNN_BiLSTM_ATTENTION_MODEL:
            return BuildCNNBiLSTMAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                                output_dim=self.output_dim)
        elif self.model_type == CNN_BiGRU_ATTENTION_MODEL:
            return BuildCNNBiGRUAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                               output_dim=self.output_dim)
        # -----------------------------Deep Hybrid + Attention models-------------------------------
        elif self.model_type == CNN_ATTENTION_LSTM_MODEL:
            return BuildCNNAttentionLSTMModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                              output_dim=self.output_dim)
        elif self.model_type == CNN_ATTENTION_GRU_MODEL:
            return BuildCNNAttentionGRUModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                             output_dim=self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiLSTM_MODEL:
            return BuildCNNAttentionBiLSTMModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                                output_dim=self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiGRU_MODEL:
            return BuildCNNAttentionBiGRUModel(trial=trial, best_params=best_params, X_train=self.X_train,
                                               output_dim=self.output_dim)
        # -----------------------------Deep More Hybrid + Attention models-------------------------------
        # elif self.model_type == CNN_ATTENTION_LSTM_ATTENTION_MODEL:
        #     return BuildCNNAttentionLSTMAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
        #                                        output_dim=self.output_dim)
        # elif self.model_type == CNN_ATTENTION_GRU_ATTENTION_MODEL:
        #     return BuildCNNAttentionGRUAttentionModel(trial=trial, best_params=best_params, X_train=self.X_train,
        #                                                output_dim=self.output_dim)
        elif self.model_type == CNN_ATTENTION_BiLSTM_ATTENTION_MODEL:
            pass
        elif self.model_type == CNN_ATTENTION_BiGRU_ATTENTION_MODEL:
            pass
        else:
            raise ValueError(
                "Invalid model type. Please choose from the available models.")
