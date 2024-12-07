import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from configparser import ConfigParser
from AI.model_pipeline import evaluate_model, train_model, load_and_prepare_data, load_config

config=ConfigParser()
config.read('config.ini')

def is_minor_version(version: str) -> bool:
    """
    Returns True if the version is a minor version (patch = 0).
    """
    parts = version.split(".")
    if len(parts) != 3:
        return False  # Invalid version format
    return parts[2] == "0"  # Keep only versions where the patch is 0


def train_and_save_models(max_versions: int = None):
    """
    Train models for each minor version and save their performance metrics to a JSON file.

    Parameters:
        max_versions (int): Maximum number of versions to process. If None, process all minor versions.

    Returns:
        None
    """
    base_dir = config["GENERAL"]["DataDirectory"]
    static_metrics_dir = config["OUTPUT"]["StaticMetricsOutputDirectory"]
    metrics_dir = os.path.join(base_dir, static_metrics_dir)
    output_dir = os.path.join(base_dir, config["OUTPUT"]["StaticModelsDirectory"])
    config_section = "VERSION"
    output_file = os.path.join(output_dir, config["MODEL"]["StaticPerformanceMetricsFile"])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize result dictionary
    results = {}

    # Get list of files and apply the minor version filter
    files = [
        file for file in os.listdir(metrics_dir)
        if file.endswith("_static_metrics.csv") and is_minor_version(file.replace("_static_metrics.csv", ""))
    ]
    if max_versions is not None:
        files = files[:max_versions]

    # Iterate over all minor versions (limited by max_versions if applicable)
    for file in files:
        version = file.replace("_static_metrics.csv", "")
        file_path = os.path.join(metrics_dir, file)
        print(f"Processing version: {version}")

        # Load configuration for the version
        param = load_config(config_section)

        # Prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, param, verbose=True)

        # Train Logistic Regression
        model_instance_lr = LogisticRegression(max_iter=5000, class_weight="balanced")
        trained_model_lr = train_model(model_instance_lr, X_train, y_train, verbose=True)
        metrics_lr = evaluate_model(trained_model_lr, X_test, y_test, verbose=True)

        # Train Random Forest
        model_instance_rf = RandomForestClassifier(class_weight="balanced")
        trained_model_rf = train_model(model_instance_rf, X_train, y_train, verbose=True)
        metrics_rf = evaluate_model(trained_model_rf, X_test, y_test, verbose=True)

        # Save results
        results[version] = {
            "LogisticRegression": metrics_lr,
            "RandomForest": metrics_rf
        }

    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
