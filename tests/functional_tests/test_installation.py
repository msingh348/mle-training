import pytest


def test_imports():
    """
    Test the import of required packages for the project.

    This test checks that all necessary packages can be imported successfully.
    If any package fails to import, the test will fail and provide an error message
    indicating which package could not be imported.

    Raises
    ------
    ImportError
        If a required package cannot be imported.
    """
    try:
        import logging
        import urllib
        import urllib.request

        import black
        import flake8
        import isort
        import matplotlib
        import numpy as np
        import pandas as pd
        import seaborn
        import sklearn

    except ImportError as e:
        pytest.fail(f"Failed to import a required package: {e}")
