import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from configparser import ConfigParser
from AI.model_pipeline import evaluate_model, train_model, load_and_prepare_data, load_config
import matplotlib.pyplot as plt
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from threading import Lock

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
    if config['MODEL'].get('SkipRetrieval', 'No').lower() == 'yes':
        print("Model training has already been done. Skipping...")
        return

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




def train_and_save_models_with_threads(source_dir: str, output_dir: str, output_file: str, max_versions: int = None):
    """
    Train models for each minor version in parallel using threads and save their performance metrics to a JSON file.

    Args:
        source_dir (str): Directory containing the metrics files.
        output_dir (str): Directory where the output file will be saved.
        output_file (str): Name of the output JSON file.
        max_threads (int): Maximum number of threads to use for processing.
        max_versions (int, optional): Maximum number of versions to process.

    Returns:
        None
    """
    
    if config['MODEL'].get('SkipRetrieval', 'No').lower() == 'yes':
        print("Model training has already been done. Skipping...")
        return

    max_threads: int = int(config["GENERAL"]["MaxThreads"])
    threads_num: int = min(max_threads, cpu_count())

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List and filter files in the source directory
    files = [
        file for file in os.listdir(source_dir)
        if file.endswith(".csv") and is_minor_version(file.split("_")[0])
    ]
    if max_versions is not None:
        files = files[:max_versions]

    results = {}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=threads_num) as executor:
        # Pass `source_dir` as an additional argument to `process_file`
        futures = {executor.submit(process_file, source_dir, file): file for file in files}
        for future in futures:
            version, metrics = future.result()
            results[version] = metrics

    # Save results to JSON
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at: {output_path}")

print_lock = Lock()

def process_file(source_dir: str, file: str):
    """
    Process a single file: Train models, evaluate them, and return metrics.

    Args:
        source_dir (str): Directory containing the metrics files.
        file (str): Name of the file to process.

    Returns:
        tuple: Version and metrics dictionary.
    """
    version = file.split("_")[0]
    file_path = os.path.join(source_dir, file)

    with print_lock:
        print(f"Processing version: {version}")

    # Load configuration for the version
    param = load_config("VERSION")

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

    return version, {
        "LogisticRegression": metrics_lr,
        "RandomForest": metrics_rf
    }


def plot_metrics_evolution(filteviolet_results: Dict):
    """
    Plot the evolution of model metrics across minor versions with improved colors and value annotations.

    Parameters:
        filteviolet_results (Dict): Dictionary with performance metrics for minor versions.

    Returns:
        None
    """
    versions = []
    auc_lr, auc_rf = [], []
    precision_lr, precision_rf = [], []
    recall_lr, recall_rf = [], []

    # Extract metrics
    for version, metrics in sorted(filteviolet_results.items(), key=lambda x: x[0]):
        versions.append(version)
        auc_lr.append(metrics["LogisticRegression"]["AUC"])
        auc_rf.append(metrics["RandomForest"]["AUC"])
        precision_lr.append(metrics["LogisticRegression"]["Precision"])
        precision_rf.append(metrics["RandomForest"]["Precision"])
        recall_lr.append(metrics["LogisticRegression"]["Recall"])
        recall_rf.append(metrics["RandomForest"]["Recall"])

    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(versions, auc_lr, label="Logistic Regression - AUC", marker="o", linestyle="-.", color="purple")
    plt.plot(versions, auc_rf, label="Random Forest - AUC", marker="s", linestyle=":", color="violet")
    annotate_points(versions, auc_lr, "purple", position="below")
    annotate_points(versions, auc_rf, "violet", position="below")
    configure_plot("AUC Evolution Across Minor Versions", "AUC")
    plt.show()

    # Plot Precision
    plt.figure(figsize=(10, 6))
    plt.plot(versions, precision_lr, label="Logistic Regression - Precision", marker="o", linestyle="-.", color="purple")
    plt.plot(versions, precision_rf, label="Random Forest - Precision", marker="s", linestyle=":", color="violet")
    annotate_points(versions, precision_lr, "purple", position="below")
    annotate_points(versions, precision_rf, "violet", position="below")
    configure_plot("Precision Evolution Across Minor Versions", "Precision")
    plt.show()

    # Plot Recall
    plt.figure(figsize=(10, 6))
    plt.plot(versions, recall_lr, label="Logistic Regression - Recall", marker="o", linestyle="-.", color="purple")
    plt.plot(versions, recall_rf, label="Random Forest - Recall", marker="s", linestyle=":", color="violet")
    annotate_points(versions, recall_lr, "purple", position="below")
    annotate_points(versions, recall_rf, "violet", position="below")
    configure_plot("Recall Evolution Across Minor Versions", "Recall")
    plt.show()

# General plot settings
def configure_plot(title: str, ylabel: str):
    plt.title(title, fontsize=14)
    plt.xlabel("Versions", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.ylim(0, 1)  # Fix y-axis between 0 and 1
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(
        loc="center left", 
        bbox_to_anchor=(1, 0.5), 
        title="Model and Metric", 
        fontsize=10
    )

# Function to annotate values below points
def annotate_points(x, y, color, position="below"):
    offset = -0.06 if position == "below" else 0.02  # Adjust position
    for i, val in enumerate(y):
        plt.text(x[i], y[i] + offset, f"{val:.2f}", ha="center", fontsize=10, color=color)
