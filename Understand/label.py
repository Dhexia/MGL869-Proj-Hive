import pandas as pd
from os import path, listdir, makedirs
from configparser import ConfigParser

# Configuration loaded once globally
config = ConfigParser()
config.read("config.ini")

def label_all_metrics(couples_df: pd.DataFrame) -> None:
    """
    Processes all metric files in the specified directory, labels files with BugStatus,
    and saves the results in the output directory.

    Parameters:
        couples_df (pd.DataFrame): DataFrame containing files with issues and their affected versions.
    """
    base_dir = config["GENERAL"]["DataDirectory"]
    metrics_directory = path.join(base_dir, config["OUTPUT"]["MetricsOutputDirectory"])
    output_dir = path.join(base_dir, config["OUTPUT"]["LabeledMetricsOutputDirectory"])
    csv_separator = config["GENERAL"].get("CSVSeparatorMetrics", ",")
    
    # Ensure the output directory exists
    if not path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        makedirs(output_dir)
    
    # Extract only filenames from the 'File' column in the couples DataFrame
    couples_df["Filename"] = couples_df["File"].apply(lambda x: path.basename(x))
    
    # Process all files in the metrics directory
    if not path.exists(metrics_directory):
        raise FileNotFoundError(f"Metrics directory not found: {metrics_directory}")
    
    for metrics_file in listdir(metrics_directory):
        if metrics_file.endswith(".csv"):
            metrics_path = path.join(metrics_directory, metrics_file)
            print(f"Processing metrics file: {metrics_path}")
            try:
                # Load metrics
                metrics_df = pd.read_csv(metrics_path, sep=csv_separator, engine="python")
                if "Kind" not in metrics_df.columns:
                    raise KeyError(f"Column 'Kind' not found in {metrics_file}")
                
                # Filter rows where Kind == "File"
                metrics_df = metrics_df[metrics_df["Kind"] == "File"]
                metrics_df["BugStatus"] = 0
                
                # Extract version from the file name
                version = metrics_file.replace("_metrics.csv", "")
                
                # Identify problematic files for the current version
                problematic_files = couples_df[
                    couples_df["Version Affected"].str.contains(version, na=False)
                ]["Filename"]
                
                # Label problematic files with BugStatus = 1
                metrics_df.loc[metrics_df["Name"].isin(problematic_files), "BugStatus"] = 1
                
                # Save labeled metrics to the output directory
                labeled_metrics_path = path.join(output_dir, f"{version}_labeled_metrics.csv")
                metrics_df.to_csv(labeled_metrics_path, index=False, sep=csv_separator)
                print(f"Labeled metrics saved to: {labeled_metrics_path}")
            except Exception as e:
                print(f"Error processing file {metrics_file}: {e}")

