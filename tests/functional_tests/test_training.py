import os
import shutil

import pandas as pd
import pytest

from src.train import train_models


@pytest.fixture
def mock_train_data(tmp_path):
    """
    Create a mock train.csv file for testing.

    This fixture generates a small mock dataset representing housing data
    and saves it as train.csv in a temporary directory.

    Parameters
    ----------
    tmp_path : pytest.tmp_path
        The temporary path provided by pytest to store the mock data.

    Returns
    -------
    str
        The file path of the generated mock train.csv.
    """
    processed_data_dir = tmp_path / "data/processed"
    os.makedirs(processed_data_dir, exist_ok=True)

    # Create a small mock dataset
    data = {
        "longitude": [-117, -98, -123, -122, -135, -145],
        "latitude": [32, 41, 34, 32, 43, 45],
        "housing_median_age": [15, 30, 45, 34, 43, 52],
        "total_rooms": [3000, 2000, 3500, 3300, 3400, 3100],
        "total_bedrooms": [100, 200, 300, 240, 280, 320],
        "population": [50, 100, 150, 200, 250, 300],
        "households": [634, 548, 745, 598, 623, 702],
        "median_income": [3, 4, 1, 2, 4, 6],
        "median_house_value": [50000, 60000, 70000, 80000, 56000, 90000],
        "ocean_proximity": [
            "NEAR OCEAN",
            "INLAND",
            "NEAR BAY",
            "NEAR H",
            "<H Bay",
            "NEAR Test",
        ],
        "income_cat": [3, 2, 5, 4, 6, 8],
    }
    mock_train_df = pd.DataFrame(data)
    mock_train_csv_path = processed_data_dir / "train.csv"

    # Save the mock dataset as train.csv
    mock_train_df.to_csv(mock_train_csv_path, index=False)

    return mock_train_csv_path


def test_train_models(tmp_path, mock_train_data):
    """
    Test the training of models using the mock train.csv.

    This test verifies that the model training function creates the expected
    model files based on the input training dataset.

    Parameters
    ----------
    tmp_path : pytest.tmp_path
        The temporary path provided by pytest to store artifacts.

    mock_train_data : str
        The file path of the mock train.csv created by the fixture.
    """
    input_file = mock_train_data  # Use the mock train.csv file
    model_output_dir = tmp_path / "artifacts"

    # Ensure the directory is empty before training
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
    os.makedirs(model_output_dir)

    train_models(input_file, model_output_dir)

    # Check if model files are created
    assert os.path.exists(os.path.join(model_output_dir, "linear_model.pkl"))
    assert os.path.exists(os.path.join(model_output_dir, "tree_model.pkl"))
    assert os.path.exists(os.path.join(model_output_dir, "best_forest_model.pkl"))
