# My ML Project

This project implements a machine learning pipeline that involves data ingestion, model training, and evaluation. It follows a structured layout using the src folder and can be easily packaged and installed as a Python library.

## Table of Contents

- Installation

- Usage

    - Data Ingestion
    - Model Training
    - Model Scoring

- Logging

- Tests

- Documentation

- Contributing

## Installation

1. Clone the repository:

```
git clone https://github.com/msingh348/mle-training.git
cd my_ml_project

```

2. Create a conda environment: The project dependencies are listed in the env.yaml file. You can create the environment using the following command:

```
conda env create -f env.yaml

```

3. Activate the environment:

```
conda activate my_ml_env
```

4. Install the project as a package: You can install the project locally as a package using `pip`:

```
pip install -e .
```


## Usage

## Data Ingestion

To download the dataset and process it for training and validation, use the ingest_data.py script:

```
python src/ingest_data.py --output-dir data/processed
```

This will save the processed data in the data/processed/ directory. By default, it downloads the dataset and splits it into training and validation sets.


## Model Training

To train the machine learning models, run the train.py script:

```
python src/train.py --input-file data/processed/train.csv --model-output-dir artifacts/models

```

This will create a artifacts/models/ directory containing trained models like `linear_model.pkl`, `tree_model.pkl`, and `best_forest_model.pkl`.


## Model Scoring

To evaluate the trained models on the validation data, use the score.py script:

```
python src/score.py --input-file data/processed/validation.csv --model-dir artifacts/models

```

The script will output performance metrics for each model.

## Logging

Logging is enabled for all scripts, and you can configure the log level, output file, and whether to display logs on the console. By default, logs are saved to the logs/ directory.

You can customize logging options by passing these arguments:

- --log-level (e.g., INFO, DEBUG)

- --log-file (to specify the log file)

- --no-console-log (to disable console logging)

```
python src/train.py --input-file data/processed/train.csv --model-output-dir artifacts/models --log-level DEBUG --log-file logs/train.log

```

## Tests

Unit and functional tests are included to verify the correctness of the code. You can run them using `pytest`:

```
pytest

```

The tests are located in the `tests/` directory and cover the following:

- `unit_tests/`: Unit tests for specific functions like data ingestion.

- `functional_tests/`: Tests to verify the overall workflow such as model training.

## Documentation:

The project uses Sphinx for documentation generation. To build the HTML documentation, follow these steps:

1. Install Sphinx (if not already installed):

```
pip install sphinx

```

2. Generate the documentation:

```
cd docs
make html

```

The generated documentation will be available in the `docs/_build/html/` directory.

## Contributing

Feel free to open issues or create pull requests if you'd like to contribute to this project.
