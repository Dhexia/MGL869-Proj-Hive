import pandas as pd
from os import path, listdir, makedirs
from configparser import ConfigParser

# Configuration loaded once globally
config = ConfigParser()
config.read("config.ini")

def label_all_metrics(couples_df: pd.DataFrame) -> None:
    """
    Processes all metric files in the specified directory, labels files with BugStatus and Priority,
    and saves the results in the output directory.

    Parameters:
        couples_df (pd.DataFrame): DataFrame containing files with issues, their affected versions, and priorities.
    """
    base_dir = config["GENERAL"]["DataDirectory"]
    metrics_directory = path.join(base_dir, config["OUTPUT"]["MetricsOutputDirectory"])
    output_dir = path.join(base_dir, config["OUTPUT"]["StaticMetricsOutputDirectory"])
    csv_separator = config["GENERAL"].get("CSVSeparatorMetrics", ",")

    if config['UNDERSTAND'].get('SkipLabelization', 'No').lower() == 'yes':
        print("Labelization process is skipped as per configuration.")
        return

    ensure_directory_exists(output_dir)
    couples_df["Filename"] = couples_df["File"].apply(lambda x: path.basename(x))

    if not path.exists(metrics_directory):
        raise FileNotFoundError(f"Metrics directory not found: {metrics_directory}")

    for metrics_file in listdir(metrics_directory):
        if metrics_file.endswith(".csv"):
            process_metrics_file(metrics_file, metrics_directory, couples_df, output_dir, csv_separator)

def ensure_directory_exists(directory: str) -> None:
    """Ensures the specified directory exists, creating it if necessary."""
    if not path.exists(directory):
        print(f"Creating output directory: {directory}")
        makedirs(directory)

def process_metrics_file(metrics_file: str, metrics_directory: str, 
                         couples_df: pd.DataFrame, output_dir: str, csv_separator: str) -> None:
    """
    Processes a single metrics file, labels it with BugStatus and numeric Priority, and saves the output.

    Parameters:
        metrics_file (str): Name of the metrics file.
        metrics_directory (str): Path to the directory containing metrics files.
        couples_df (pd.DataFrame): DataFrame with affected files, versions, and priorities.
        output_dir (str): Directory to save the labeled metrics.
        csv_separator (str): CSV separator to use for reading/writing files.
    """
    # Mapping priorities to numeric values
    priority_mapping = {
        "None": 0,
        "Trivial": 1,
        "Minor": 2,
        "Major": 3,
        "Critical": 4,
        "Blocker": 5
    }

    metrics_path = path.join(metrics_directory, metrics_file)
    print(f"Processing metrics file: {metrics_path}")

    try:
        metrics_df = pd.read_csv(metrics_path, sep=csv_separator, engine="python")
        if "Kind" not in metrics_df.columns:
            raise KeyError(f"Column 'Kind' not found in {metrics_file}")

        metrics_df = metrics_df[
            (metrics_df["Kind"] == "File") &
            (metrics_df["Name"].str.endswith((".java", ".cpp")))
        ]
        metrics_df["BugStatus"] = 0
        metrics_df["Priority"] = 0  # Default numeric value for Priority (None -> 0)

        version = metrics_file.replace("_metrics.csv", "")
        problematic_files = couples_df[couples_df["Version Affected"].str.contains(version, na=False)]

        # Update BugStatus and Priority for problematic files
        for _, row in problematic_files.iterrows():
            file_name = row["Filename"]
            priority = row["Priority"]

            metrics_df.loc[metrics_df["Name"] == file_name, "BugStatus"] = 1
            metrics_df.loc[metrics_df["Name"] == file_name, "Priority"] = priority_mapping.get(priority, 0)

        labeled_metrics_path = path.join(output_dir, f"{version}_labeled_metrics.csv")
        metrics_df.to_csv(labeled_metrics_path, index=False, sep=csv_separator)
        print(f"Labeled metrics saved to: {labeled_metrics_path}")
    except Exception as e:
        print(f"Error processing file {metrics_file}: {e}")





