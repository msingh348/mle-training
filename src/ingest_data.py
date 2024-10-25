import argparse
import logging
import os
import tarfile
import urllib
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import setup_logging

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path="data/raw"):
    """
    Download and extract housing data.

    Parameters
    ----------
    housing_url : str, optional
        The URL from where to download the housing data (default is `HOUSING_URL`).
    housing_path : str, optional
        The directory to save the downloaded and extracted data (default is `"data/raw"`).

    Returns
    -------
    None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logging.info(f"Housing data downloaded and extracted to {housing_path}")


def load_housing_data(housing_path="data/raw"):
    """
    Load housing data from the specified directory.

    Parameters
    ----------
    housing_path : str, optional
        The directory containing the housing data CSV file (default is `"data/raw"`).

    Returns
    -------
    pd.DataFrame
        The loaded housing data as a Pandas DataFrame.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    logging.info(f"Loading housing data from {csv_path}")
    return pd.read_csv(csv_path)


def clean_data(housing):
    """
    Clean the housing dataset by adding an income category feature.

    Parameters
    ----------
    housing : pd.DataFrame
        The housing data as a Pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The housing data with an additional 'income_cat' feature.
    """
    logging.info("Cleaning data by creating income categories.")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def create_train_val_datasets(output_dir):
    """
    Create training and validation datasets and save them as CSV files.

    The housing data is downloaded, cleaned, and split into training and validation sets.

    Parameters
    ----------
    output_dir : str
        The directory to save the train and validation datasets.

    Returns
    -------
    None
    """
    fetch_housing_data()
    housing = load_housing_data()
    housing = clean_data(housing)
    train_set, val_set = train_test_split(housing, test_size=0.2, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_set.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_set.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    logging.info(f"Training and validation datasets saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest data and create train/validation datasets."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",  # Allows the argument to be optional
        default="data/processed",  # Default value for output_dir
        help="Output directory for the datasets",
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

    create_train_val_datasets(args.output_dir)
