import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from utils import setup_logging


def add_features(housing):
    """
    Add new features to the housing dataset.

    This function creates three new features: 'rooms_per_household',
    'bedrooms_per_room', and 'population_per_household' based on
    the existing data in the housing dataset.

    Parameters
    ----------
    housing : pd.DataFrame
        The housing dataset to which the new features are added.

    Returns
    -------
    pd.DataFrame
        The housing dataset with the new features added.
    """
    logging.info("Adding new features to dataset.")
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing


def train_models(input_file, model_output_dir):
    """
    Train multiple machine learning models and save them to disk.

    This function trains three models: a linear regression model, a decision
    tree regressor, and a random forest regressor with hyperparameter tuning.
    The trained models are saved as pickle files in the specified output directory.

    Parameters
    ----------
    input_file : str
        Path to the CSV file containing the training dataset.
    model_output_dir : str
        Directory where the trained models will be saved as pickle files.

    Returns
    -------
    None
    """
    logging.info(f"Starting training process with dataset {input_file}")
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        housing = pd.read_csv(input_file)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    housing = add_features(housing)
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)
    housing = pd.get_dummies(housing, drop_first=True)

    numeric_cols = housing.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    housing[numeric_cols] = imputer.fit_transform(housing[numeric_cols])

    # Train and save models
    logging.info("Training Linear Regression model.")
    lin_reg = LinearRegression()
    lin_reg.fit(housing, housing_labels)
    with open(f"{model_output_dir}/linear_model.pkl", "wb") as f:
        pickle.dump(lin_reg, f)

    logging.info("Training Decision Tree model.")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing, housing_labels)
    with open(f"{model_output_dir}/tree_model.pkl", "wb") as f:
        pickle.dump(tree_reg, f)

    logging.info("Training Random Forest with hyperparameter tuning.")
    param_distribs = {
        "n_estimators": np.arange(10, 200, 10),
        "max_features": ["sqrt", "log2"],
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing, housing_labels)
    with open(f"{model_output_dir}/best_forest_model.pkl", "wb") as f:
        pickle.dump(rnd_search.best_estimator_, f)

    logging.info(f"Models saved in {model_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models and save them.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Input dataset file path",
        default="data/processed/train.csv",  # default path
        nargs="?",
    )  # This makes the argument optional
    parser.add_argument(
        "model_output_dir",
        type=str,
        help="Output directory for the model pickles",
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

    train_models(args.input_file, args.model_output_dir)
