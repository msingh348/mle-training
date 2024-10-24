import os
import sys

import pytest

from src.ingest_data import fetch_housing_data, load_housing_data

# Add the src directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


@pytest.fixture
def setup_output_dir(tmp_path):
    """
    Create a temporary output directory for raw data.

    This fixture sets up a temporary directory to hold raw data files
    during testing. The directory will be automatically cleaned up
    after the test completes.

    Parameters
    ----------
    tmp_path : pytest.tmp_path
        The temporary path provided by pytest to store raw data.

    Returns
    -------
    pathlib.Path
        The path to the temporary output directory created for tests.
    """
    output_dir = tmp_path / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_fetch_housing_data():
    """
    Test the fetch_housing_data function.

    This test checks whether the housing data is successfully downloaded
    and extracted to the specified output directory.

    Parameters
    ----------
    setup_output_dir : pathlib.Path
        The path to the temporary directory created by the fixture.
    """
    output_dir = "data/raw"  # Adjust as needed
    fetch_housing_data(housing_path=output_dir)

    # Check if the data was downloaded and extracted
    assert os.path.exists(os.path.join(output_dir, "housing.csv"))


def test_load_housing_data():
    """
    Test the load_housing_data function.

    This test verifies that the load_housing_data function correctly loads
    the housing data from the specified path and checks that it is not empty.

    Parameters
    ----------
    setup_output_dir : pathlib.Path
        The path to the temporary directory created by the fixture.
    """
    # Assuming the file exists after fetch_housing_data
    housing = load_housing_data(housing_path="data/raw")
    assert housing is not None
    assert not housing.empty
