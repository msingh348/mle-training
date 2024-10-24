from setuptools import find_packages, setup

setup(
    name="mle-training",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pytest",
        "black",
        "isort",
        "flake8",
        "sphinx",
    ],
    entry_points={
        "console_scripts": [
            "ingest_data=ingest_data:create_train_val_datasets",  # Update to correct function path
            "train=train:train_models",
            "score=score:score_models",
        ],
    },
)
