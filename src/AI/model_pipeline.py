import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from typing import Dict, Callable, Tuple
from sklearn.base import BaseEstimator
from configparser import ConfigParser


config=ConfigParser()
config.read('config.ini')

# 1. Load configuration for a specific version or scenario
def load_config(config_section: str) -> Dict[str, any]:
    """
    Loads the configuration parameters for a specific section.
    
    :param config_section: Section name in the config.ini file.
    :return: Dictionary containing configuration parameters.
    """

    if config_section not in config:
        raise ValueError(f"Section '{config_section}' not found in config.ini.")
    
    model_config = {
        "target_column": config[config_section]["target_column"],
        "drop_columns": config[config_section]["drop_columns"].split(","),
        "test_size": float(config[config_section].get("test_size", 0.3)),
        "random_state": int(config[config_section].get("random_state", 42)),
    }
    return model_config


# 2. Load and prepare data
def load_and_prepare_data(
    file_path: str,
    config: Dict[str, any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads data, cleans it, and prepares training and testing sets.
    
    :param file_path: Path to the CSV file.
    :param config: Configuration dictionary containing model parameters.
    :return: X_train, X_test, y_train, y_test
    """
    # Load data with the specified separator
    df = pd.read_csv(file_path, sep=",")

    # Replace commas in numerical columns with dots for proper float conversion
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

    # Convert all possible columns to numeric, ignoring errors for non-numeric columns
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop unused columns
    drop_columns = config.get("drop_columns", [])
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Separate features (X) and target (y)
    target_column = config["target_column"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get("test_size", 0.3), random_state=config.get("random_state", 42)
    )

    return X_train, X_test, y_train, y_test



# 3. Train model
def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> BaseEstimator:
    """
    Trains a given model on the provided data.
    """
    model.fit(X_train, y_train)
    return model


# 4. Evaluate model
def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates a model's performance on test data.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }

    return metrics


# 5. Pipeline
def run_pipeline(
    file_path: str,
    model: Callable[[], BaseEstimator],
    config_section: str
) -> Dict[str, float]:
    """
    Runs the entire pipeline for training and evaluating a model.
    
    :param file_path: Path to the CSV file.
    :param model: Callable that returns an instance of a scikit-learn model.
    :param config_section: Name of the section in the config.ini file to load parameters from.
    :return: Dictionary of evaluation metrics.
    """
    # Load configuration
    config = load_config(config_section)
    
    # Prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, config)
    
    # Train model
    model_instance = model()
    trained_model = train_model(model_instance, X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(trained_model, X_test, y_test)

    return metrics
