import random

random.seed(42)  # define seed

import numpy as np

np.random.seed(42)  # define seed

import tensorflow as tf

tf.random.set_seed(42)  # define seed

# Reduce randomness due the GPU manipulation
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.config.optimizer.set_jit(False)
tf.config.experimental.list_physical_devices("GPU")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the desired GPU device
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import warnings

warnings.filterwarnings("ignore")
import gc
import keras

keras.utils.set_random_seed(42)
import pandas as pd
from os.path import join
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import save_model
from tensorflow.keras.initializers import glorot_normal
from tensorflow import keras
from scipy.optimize import minimize
from utils import (
    get_processed_data,
    create_model,
    save_best_params,
    is_not_require_params,
    train_model,
    calculate_weighted_mean,
    evaluate_model,
    get_model_preds,
    delete_file,
    empty_directory,
)


def find_best_hyperparameter(
    train_inputs,
    test_inputs,
    train_output,
    test_output,
    best_model_path,
    best_hyperparameter_path,
    total_iterations=100,
    skip_iterations=0,
):
    """
    This function performs hyperparameter tuning.

    Parameters:
    train_inputs (numpy array): training input data
    test_inputs (numpy array): testing input data
    train_output (numpy array): training output data
    test_output (numpy array): testing output data
    best_model_path (string): path to save the best model
    best_hyperparameter_path (string): path to save the best hyperparameters

    Returns:
    best_hyperparameter (dict): Dictionary of best hyperparameter.

    """
    # Process train and test dataset
    train_X, train_Y = get_processed_data(train_inputs, train_output)
    test_X, test_Y = get_processed_data(test_inputs, test_output)
    n_timesteps, n_features = train_X.shape[1], train_X.shape[2]

    # Define the hyperparameter search space
    PARAM_SPACE = {
        "filters_1": list(range(2, 15, 2)),
        "filters_2": list(range(4, 25, 2)),
        "filters_3": list(range(8, 35, 2)),
        "kernel_size_1": list(range(3, 19, 2)),
        "kernel_size_2": list(range(5, 17, 2)),
        "kernel_size_3": list(range(7, 15, 2)),
        "dense_units_1": [64, 128, 256, 512],
        "dense_units_2": [8, 16, 32, 64, 128, 256],
        "dropout_rate": [0.10, 0.2, 0.3, 0.4, 0.5],
        "optimizer": ["nadam", "adam", "rmsprop"],
        "batch_size": [8, 16, 24, 32, 64, 128],
    }

    # To avoid processing duplicate parameters
    processed_params = []
    best_r2 = 0
    best_hyper_params = None
    iteration = 0

    while iteration < total_iterations:
        # Randomly select parameters from the parameter space
        hyper_params = {
            key: np.random.choice(value) for key, value in PARAM_SPACE.items()
        }

        # Skip if parameters are already processed or not required
        if hyper_params in processed_params or is_not_require_params(hyper_params):
            continue

        iteration += 1
        # Mark parameters as processed
        processed_params.append(hyper_params)

        if iteration < skip_iterations:
            continue

        print(f"Iteration-{iteration}...")

        # Create and train model
        model = create_model(n_timesteps, n_features, **hyper_params)
        model = train_model(model, train_X, train_Y, test_X, test_Y, hyper_params)
        y_pred = model.predict(test_X).reshape(-1)
        curr_r2 = round(r2_score(test_Y, y_pred), 2)

        # Update best parameters if current R2 score is better
        if curr_r2 > best_r2:
            best_r2, best_hyper_params = curr_r2, hyper_params
            save_best_params(best_hyperparameter_path, best_hyper_params, best_r2)
            save_model(model, best_model_path)
            print(f"New best R2 score: {best_r2}")
            print(f"New Best hyperparameters: {best_hyper_params}")

    print("Hyperparameter tuning completed.")
    return best_hyper_params


def start_model_training(
    train_inputs,
    test_inputs,
    train_output,
    test_output,
    total_models,
    model_dir,
    hyper_params,
):
    """
    Train multiple models, create an ensemble, and evaluate its performance.

    Parameters:
        train_inputs (list): List of training input data.
        test_inputs (list): List of testing input data.
        train_output (numpy.ndarray): Training output data.
        test_output (numpy.ndarray): Testing output data.
        total_models (int): Total number of models to train.
        model_dir (str): Path to the directory containing trained models.
        hyper_params (dict): Dictionary containing hyperparameters for model training.

    Returns:
        tuple: A tuple containing weighted average predictions and ensemble output.
            - weighted_avg_pred (numpy.ndarray): Weighted average predictions.
            - ensemble_output (float): Ensemble output evaluation result.
    """
    # Process train and test dataset
    train_X, train_Y = get_processed_data(train_inputs, train_output)
    test_X, test_Y = get_processed_data(test_inputs, test_output)
    n_timesteps, n_features = train_X.shape[1], train_X.shape[2]

    # Getting the CNN model architecture
    model = create_model(n_timesteps, n_features, **hyper_params)

    # Train and get model predictions
    model_preds = get_model_preds(
        model_dir, hyper_params, total_models, model, train_X, train_Y, test_X, test_Y
    )

    # Calculate weighted mean model predictions
    weighted_avg_pred = calculate_weighted_mean(model_preds, test_Y)
    # print("Weighted Average Predictions: ", weighted_avg_pred)

    # Output predictions
    ensemble_output = evaluate_model(weighted_avg_pred, test_Y)
    print(f"Model score: {ensemble_output}")

    return weighted_avg_pred
