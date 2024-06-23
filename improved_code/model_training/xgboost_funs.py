from sklearn.metrics import r2_score, mean_squared_error
from hyperopt import fmin, tpe, rand, hp, Trials
from scipy import stats
from scipy.optimize import minimize
from constants import XGBOOST_MODELS_PATH
import numpy as np
import pickle
import pandas as pd
from os.path import join
import warnings
import xgboost as xgb
import os


warnings.filterwarnings("ignore")


# Loading training and test data:
train_indices = list(
    np.load(
        join("..", "..", "data", "kcat_data", "splits", "CV_train_indices.npy"),
        allow_pickle=True,
    )
)
test_indices = list(
    np.load(
        join("..", "..", "data", "kcat_data", "splits", "CV_test_indices.npy"),
        allow_pickle=True,
    )
)


def delete_existing_file(path):
    if os.path.exists(path):
        print("file already exists. Deleting existing file...")
        os.remove(path)


def save_pickel_model(model, model_name):
    """Saving Xgboost Models"""
    model_path = join(XGBOOST_MODELS_PATH, model_name)
    
    delete_existing_file(model_path)

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)


def save_best_params(params_path, params, score="UNK"):
    """
    Save the best hyperparameters and their corresponding score to a file.

    Parameters:
    - params (dict): Dictionary containing hyperparameter names and values.
    - score (float): The score associated with the hyperparameters.

    Returns:
    None
    """

    delete_existing_file(params_path)

    with open(params_path, "w") as file:
        params_info = str(score) + "\n" + str(params)
        file.write(str(params_info))


def get_processed_data(input_feats, out):
    """
    Process input data.

    Parameters:
    - input_feats (list): Input data.
    - out (list): Output data.

    Returns:
    - X (numpy.ndarray): Processed input features.
    - y (numpy.ndarray): Processed target variable.
    """
    if len(input_feats) > 1:
        input_feats = [np.array(list(feat)) for feat in input_feats]
        X = np.concatenate(input_feats, axis=1)
    else:
        X = np.array(list(input_feats[0]))
    y = np.array(list(out))

    return X, y


"""
def cross_validation_mse_gradient_boosting(param, train_X, train_Y, test_X, test_Y):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"

    MSE = []
    R2 = []
    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(train_X[train_index], label=train_Y[train_index])
        dvalid = xgb.DMatrix(train_X[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)
        y_valid_pred = bst.predict(dvalid)
        MSE.append(
            np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2)
        )
        R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
    mean_r2_score = np.mean(R2)
    print("R2 score: ", mean_r2_score)
    return -mean_r2_score
"""


def cross_validation_mse_gradient_boosting(param, train_X, train_Y):
    num_round = param["num_rounds"]
    del param["num_rounds"]
    param["max_depth"] = int(np.round(param["max_depth"]))
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"

    MSE = []
    R2 = []
    for i in range(5):
        train_index, test_index = train_indices[i], test_indices[i]
        dtrain = xgb.DMatrix(train_X[train_index], label=train_Y[train_index])
        dvalid = xgb.DMatrix(train_X[test_index])
        bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)
        y_valid_pred = bst.predict(dvalid)
        MSE.append(
            np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2)
        )
        R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
    mean_r2_score = np.mean(R2)
    return mean_r2_score


def train_xgboost(train_X, train_Y, test_X, test_Y, param, model_name):
    # Creating input matrices:
    print(f"XGBOOST training of {model_name}")
    train_X, train_Y = get_processed_data(train_X, train_Y)
    test_X, test_Y = get_processed_data(test_X, test_Y)

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X)

    num_round = int(param["num_rounds"])
    param["max_depth"] = int(np.round(param["max_depth"]))
    del param["num_rounds"]

    bst = xgb.train(param, dtrain, num_round, verbose_eval=False)
    y_test_pred = bst.predict(dtest)

    MSE_dif_fp_test = mean_squared_error(np.reshape(test_Y, (-1)), y_test_pred)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

    output = {
        "pearsonr": np.round(Pearson[0], 2),
        "r2": np.round(R2_dif_fp_test, 2),
        "mse": np.round(MSE_dif_fp_test, 2),
    }
    print(f"{model_name} results: {output}")
    save_pickel_model(bst, f"{model_name}.h5")


"""
def find_best_params(train_X, train_Y, test_X, test_Y, params_file, max_evals=200):
    train_X, train_Y = get_processed_data(train_X, train_Y)
    test_X, test_Y = get_processed_data(test_X, test_Y)

    space_gradient_boosting = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 1),
        "max_depth": hp.uniform("max_depth", 4, 12),
        # "subsample": hp.uniform("subsample", 0.7, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "max_delta_step": hp.uniform("max_delta_step", 0, 5),
        "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
        "num_rounds": hp.uniform("num_rounds", 20, 200),
    }

    trials = Trials()
    best = fmin(
        fn=lambda params: cross_validation_mse_gradient_boosting(
            params, train_X, train_Y, test_X, test_Y
        ),
        space=space_gradient_boosting,
        algo=rand.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    save_best_params(f"../../hyperparameters/{params_file}", best)
    print("Best params: ", best)
"""


def find_best_params(train_X, train_Y, test_X, test_Y, params_file, max_evals=200):
    train_X, train_Y = get_processed_data(train_X, train_Y)
    test_X, test_Y = get_processed_data(test_X, test_Y)

    space_gradient_boosting = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 1),
        "max_depth": hp.uniform("max_depth", 4, 12),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "max_delta_step": hp.uniform("max_delta_step", 0, 5),
        "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
        "num_rounds": hp.uniform("num_rounds", 20, 200),
    }

    trials = Trials()
    best_r2_score = float("-inf")
    best_hyperparams = None

    def objective(params):
        nonlocal best_r2_score, best_hyperparams
        params_orig = params.copy()
        r2_score = cross_validation_mse_gradient_boosting(params, train_X, train_Y)
        if r2_score > best_r2_score:
            best_r2_score = r2_score
            best_hyperparams = params_orig
            new_hyp_file = os.path.join(HYPERPARAMETERS_PATH, params_file)

            save_best_params(
                new_hyp_file, best_hyperparams, best_r2_score
            )
        return -r2_score

    best = fmin(
        fn=objective,
        space=space_gradient_boosting,
        algo=rand.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    print("Best hyperparameters:", best)


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
    r2 = r2_score(test_Y, pred_Y)
    output = {
        "mse": round(mse, 2),
        "R2 score": round(r2, 2),
        "pearson coefficient": round(np.sqrt(r2), 2),
    }

    return output


def load_pickle_model(model_name):
    with open(model_name, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def infer_xgboost(test_X, test_Y, model_name):
    test_X, test_Y = get_processed_data(test_X, test_Y)
    dtest = xgb.DMatrix(test_X)
    bst = load_pickle_model(model_name)
    y_test_pred = bst.predict(dtest)
    return y_test_pred
