import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

from utils import setup_logging


def load_data(data_file):
    """
    loading data from the specified file

    Parameters
    ----------
    data_file : str, optional
        Path to the csv file containing the validation data (default is `"data/processed/val.csv"`).

    Returns
    -------
    pd.DataFrame
        The validation data as a pandas data frame.
    """
    logging.info(f"Loading validation data from {data_file}")
    return pd.read_csv(data_file)


def load_model(model_file):
    """
    Load a machine learning model from a file.

    Parameters
    ----------
    model_file : str
        Path to the file containing the saved model in pickle format.

    Returns
    -------
    object
        The loaded machine learning model.
    """
    logging.info(f"Loading model from {model_file}")
    with open(model_file, "rb") as f:
        return pickle.load(f)


def preprocess_data(housing):
    """
    Preprocess the housing data by engineering additional features.

    This function adds three new features to the housing data:
    'rooms_per_household', 'bedrooms_per_room', and 'population_per_household'.
    It also drops the 'median_house_value' column and performs one-hot encoding.

    Parameters
    ----------
    housing : pd.DataFrame
        The housing data to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed housing data.
    """
    logging.info("Preprocessing validation data.")
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = pd.get_dummies(housing, drop_first=True)
    housing = housing.drop("median_house_value", axis=1, errors="ignore")

    return housing


def score_models(input_file, model_dir):
    """
    Score multiple models using the validation dataset.

    This function loads the validation data and processes it. It loads several
    machine learning models from the specified directory, makes predictions, and
    calculates the RMSE for each model.

    Parameters
    ----------
    input_file : str
        Path to the CSV file containing the validation data.
    model_dir : str
        Directory containing the saved machine learning models.

    Returns
    -------
    None
    """
    housing_val = load_data(input_file)
    housing_val_prepared = preprocess_data(housing_val)

    imputer = SimpleImputer(strategy="median")
    housing_val_prepared = imputer.fit_transform(housing_val_prepared)

    logging.info("Scoring models.")
    models = ["linear_model.pkl", "tree_model.pkl", "best_forest_model.pkl"]
    for model_name in models:
        model = load_model(f"{model_dir}/{model_name}")
        predictions = model.predict(housing_val_prepared)
        rmse = root_mean_squared_error(housing_val["median_house_value"], predictions)
        logging.info(f"RMSE for {model_name}: {rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score models using validation dataset."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Validation dataset file path",
        default="data/processed/val.csv",  # default path
        nargs="?",
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory with trained models",
        default="models/",  # output directory
        nargs="?",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )

    args = parser.parse_args()
    setup_logging(args.log_level, args.log_file, not args.no_console_log)

    score_models(args.input_file, args.model_dir)
