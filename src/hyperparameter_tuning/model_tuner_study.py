import optuna
import json
import os
import time

from hyperparameter_tuning.model_tuner import ModelTuner
from input_processing.data_processing import preprocess_augment_and_split_dataset
from output_processing.custom_functions import (evaluate_model, plot_losses, plot_evaluation_metrics,
                                                save_evaluation_metrics, save_loss_to_txt,
                                                predict_next_x_days, save_trained_model,
                                                load_trained_model, plot_predictions, save_best_params)
from utils.constants import (BASE_PATH, N_TRIAL, HYPERBAND_PATH, ELECTRICITY)
import logging

logger = logging.getLogger(__name__)


def model_tuner_and_study(look_backs, forecast_periods, model_types, series_type):
    for _ser in series_type:
        for look_back_day in look_backs:
            for forecast_day in forecast_periods:
                logger.info(
                    f"Tuning with series_type={_ser} | look_back={look_back_day} | forecast_period={forecast_day}")
                X_train, X_val, y_train, y_val, _ = preprocess_augment_and_split_dataset(ELECTRICITY, 'D',
                                                                                         look_back_day,
                                                                                         forecast_day)

                for model in model_types:
                    path = f"{BASE_PATH + _ser}/{HYPERBAND_PATH + model}/{look_back_day}_{forecast_day}_best_params.json"
                    start_time = time.time()
                    model_tuner = ModelTuner(X_train, y_train, X_val, y_val, forecast_day, model_types[0])

                    # Optuna study
                    study = optuna.create_study(direction='minimize')
                    study.optimize(model_tuner.objective, n_trials=N_TRIAL)

                    logger.info("Best trial:")
                    logger.info(study.best_trial.params)

                    # best hyperparameters
                    best_params = study.best_trial.params
                    end_time = time.time()
                    total_time = end_time - start_time
                    save_best_params(path, model, best_params, total_time)

    return model_tuner, study, best_params
