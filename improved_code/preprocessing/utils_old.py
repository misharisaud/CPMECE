import tensorflow as tf
tf.random.set_seed(42)  # define seed

import numpy as np
np.random.seed(42)  # define seed

import os
import shutil

from tensorflow import keras
from tensorflow.keras.initializers import glorot_normal
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize


def reshape_dims(inp_X):
    """
    Reshape input data for CNN model.

    Parameters:
    - inp_X (numpy.ndarray): Input data.

    Returns:
    - inp_reshaped (numpy.ndarray): Reshaped input data for the CNN model.
    """
    sample_size = inp_X.shape[0]  # number of samples
    time_steps = inp_X.shape[1]  # number of features
    input_dimension = 1  # each feature is represented by 1 number
    inp_reshaped = inp_X.reshape(sample_size,
                                 time_steps,
                                 input_dimension)
    return inp_reshaped


def get_processed_data(inp_f1, inp_f2, out):
    """
    Process input data.

    Parameters:
    - data (pandas.DataFrame): Input data.

    Returns:
    - X (numpy.ndarray): Processed input features.
    - y (numpy.ndarray): Processed target variable.
    """
    X = np.concatenate(
        [
            np.array(list(inp_f1)),
            np.array(list(inp_f2))
        ],
        axis=1
    )
    X = reshape_dims(X)
    y = np.array(list(out))

    return X, y


def create_model(
    n_timesteps, 
    n_features,
    filters_1,
    filters_2,
    filters_3,
    kernel_size_1,
    kernel_size_2,
    kernel_size_3,
    dense_units_1,
    dense_units_2,
    dropout_rate,
    optimizer,    
    batch_size=None,
):
    """
    Create a Convolutional Neural Network (CNN) model.

    Parameters:
    - n_timesteps (int): Number of time steps in input sequence.
    - n_features (int): Number of features in input sequence.
    - filters_1, filters_2, filters_3 (int): Number of filters for convolutional layers.
    - kernel_size_1, kernel_size_2, kernel_size_3 (float): Size of convolutional kernels.
    - dense_units_1, dense_units_2 (int): Number of units in fully connected layers.
    - dropout_rate (float): Dropout rate for regularization.
    - optimizer (str): Name of the optimizer.
    - batch_size (int): Batch size for training.
    - weight_initializer (str): Weight initializer for the model.

    Returns:
    - model (keras.Sequential): Compiled CNN model.
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))

    # conv + max layer 1
    model.add(
        keras.layers.Conv1D(
            filters=filters_1,
            kernel_size=int(kernel_size_1),
            activation="relu",
            kernel_initializer=glorot_normal(seed=42)
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    # conv + max layer 2
    model.add(
        keras.layers.Conv1D(
            filters=filters_2,
            kernel_size=int(kernel_size_2),
            activation="relu",
            kernel_initializer=glorot_normal(seed=42)
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    # conv + max layer 3
    model.add(
        keras.layers.Conv1D(
            filters=filters_3,
            kernel_size=int(kernel_size_3),
            activation="relu",
            kernel_initializer=glorot_normal(seed=42)
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    # fully connected layers
    model.add(keras.layers.Flatten())

    # FC L1
    model.add(keras.layers.Dense(dense_units_1,
                                 activation="relu",
                                 kernel_initializer=glorot_normal(seed=42)))

    model.add(keras.layers.Dropout(dropout_rate))
    # FC L2
    model.add(keras.layers.Dense(dense_units_2,
                                 activation="relu",
                                 kernel_initializer=glorot_normal(seed=42)))
    # output layer
    model.add(keras.layers.Dense(n_features,
                                 activation="linear",
                                 kernel_initializer=glorot_normal(seed=42)))

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model


def save_best_params(params_path, params, score):
    '''
    Save the best hyperparameters and their corresponding score to a file.

    Parameters:
    - params (dict): Dictionary containing hyperparameter names and values.
    - score (float): The score associated with the hyperparameters.

    Returns:
    None
    '''
    with open(params_path, "w") as file:
        file.write(f"Best Score: {score}\n")
        for key, value in params.items():
            file.write(f"{key}: {value}\n")


def delete_file(file_path):
    """
    Delete a file if it exists.

    Parameters:
    - file_path (str): Path to the file to be deleted.

    Returns:
    - None
    """
    if os.path.isfile(file_path):
        os.unlink(file_path)


def empty_directory(directory_path):
    """
    Empty the contents of a directory.

    Parameters:
    - directory_path (str): Path to the directory to be emptied.

    Returns:
    - None
    """
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error: {e}")


def is_not_require_params(params):
    '''
    Check if certain hyperparameters do not meet specific conditions for great results.

    Parameters:
    - params (dict): Dictionary containing hyperparameter names and values.

    Returns:
    bool: True if the given hyperparameters do not meet the conditions, otherwise False.
    '''
    # Conditions for ignoring new hyperparameters
    neurons_cond = params["dense_units_2"] > params["dense_units_1"]
    filters_cond = (
        params["filters_2"] < params["filters_1"]
        or params["filters_3"] < params["filters_2"]
        or params["filters_3"] < params["filters_1"]
    )

    return neurons_cond or filters_cond


def train_model(model, train_X, train_Y, test_X, test_Y, params):
    """
    Train the neural network model.

    Parameters:
    - model (keras.Sequential): Compiled neural network model.
    - train_X (numpy.ndarray): Input features for training.
    - train_Y (numpy.ndarray): Target variable for training.
    - test_X (numpy.ndarray): Input features for validation.
    - test_Y (numpy.ndarray): Target variable for validation.
    - params (dict): Dictionary containing training parameters.

    Returns:
    - model (keras.Sequential): Trained neural network model.
    """
    # scheduling EarlyStopping callback
    early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=0,
            restore_best_weights=True,
            mode="min",
        )
    # learning rate scheduling using the ReduceLROnPlateau callback
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0,
        mode="min",
    )
    model.fit(train_X,
              train_Y,
              epochs=60,
              callbacks=[early_stopping, lr_scheduler],
              batch_size=params["batch_size"],
              verbose=0,
              validation_data=(test_X, test_Y)
              )

    return model


def evaluate_models_weight(weights, model_preds, true_values):
    """
    Evaluate models based on weighted average R2 score.

    Parameters:
    - weights (numpy.ndarray): Array of weights for each model.
    - model_preds (numpy.ndarray): Predictions from each model.
    - true_values (numpy.ndarray): True target values.

    Returns:
    - r2 (float): Weighted average R2 score (negative, as it is minimized).
    """
    weighted_avg = np.average(model_preds, weights=weights, axis=0)
    r2 = r2_score(true_values, weighted_avg)
    return -r2  # We want to maximize R2, so minimize -R2


def calculate_weighted_mean(model_preds, true_values):
    """
    Calculate the weighted mean of model predictions.

    Parameters:
    - model_preds (numpy.ndarray): Predictions from each model.
    - true_values (numpy.ndarray): True target values.

    Returns:
    - weighted_avg_pred (numpy.ndarray): Weighted average of model predictions.
    """
    num_models = model_preds.shape[0]
    # Initial weights (for illustration, you can adjust this as needed)
    initial_weights = np.ones(num_models) / num_models
    # Constraint: the sum of weights must be 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds for weights (between 0 and 1)
    bounds = [(0, 1)] * num_models

    result = minimize(
        evaluate_models_weight,
        initial_weights,
        args=(model_preds, true_values),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = result.x
    weighted_avg_pred = np.average(model_preds, weights=best_weights, axis=0)
    return weighted_avg_pred


def evaluate_model(pred_Y, test_Y):
    """
    Evaluate the performance of the model.

    Parameters:
    - pred_Y (numpy.ndarray): Predicted values.
    - test_Y (numpy.ndarray): True target values.
    """
    mse = mean_squared_error(test_Y, pred_Y)
    r2 = r2_score(test_Y,  pred_Y)
    output = {
        "mse": round(mse, 2),
        "R2 score": round(r2, 2),
        "pearson coefficient": round(np.sqrt(r2), 2)
        }
        
    return output


def get_model_preds(models_dir,
                    hyper_params, 
                    total_models,
                    model, 
                    train_X, 
                    train_Y, 
                    test_X, 
                    test_Y):
    """
    Get predictions from an ensemble of models.

    Parameters:
    - models_dir (str): Saved models directory path
    - hyper_params (dict): Contain trained Hyperparameters
    - total_models (int): total models trained for ensemble (to reduce the randomness)
    - model (keras.Sequential): The base model.
    - train_X (numpy.ndarray): Training input features.
    - train_Y (numpy.ndarray): Training target variable.
    - test_X (numpy.ndarray): Testing input features.
    - test_Y (numpy.ndarray): Testing target variable.

    Returns:
    - model_preds (numpy.ndarray): Predictions from the ensemble of models.
    """
    empty_directory(models_dir)
    # define callback for early stopping to avoid overfitting.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=0,
        restore_best_weights=True,
        mode="min",
    )

    # define callback for updating learning rate to get the optimized point
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0,
        mode="min",
    )
    # ensemble the models to overcome the randomness
    model_preds = []
    for i in range(total_models):
        # model_path = os.path.join(MODEL_DIR, f"{i}.h5")
        # start training
        model.fit(
            train_X,
            train_Y,
            epochs=60,
            callbacks=[early_stopping, lr_scheduler],
            batch_size=hyper_params["batch_size"],
            verbose=0,
            validation_data=(test_X, test_Y),
            shuffle=False,  # Set shuffle to False
        )

        # save_model(model, model_path)  # saved model
        pred_Y = model.predict(test_X)  # predicting            
        model_preds.append(pred_Y)
        print(f"Model-{i+1} results {evaluate_model(pred_Y, test_Y)}")
        

    return np.array(model_preds)
