
import pandas as pd
import os
from os.path import exists, join
from configparser import ConfigParser
from os import makedirs

config = ConfigParser()
config.read("config.ini")

def merge_static_and_dynamic_csv():
    """
    Merge static and dynamic metrics CSVs for each version, saving results in the specified directory.

    The function reads:
        - The directory containing the static metrics CSVs from the configuration file.
        - The directory containing the dynamic metrics CSVs.
        - The output directory for saving the merged CSVs.
    
    Returns:
        None
    """

    # Load configuration from config.ini ----------------------------------- #
    data_directory = config["GENERAL"]["DataDirectory"]
    static_metrics_subdir = config["OUTPUT"]["LabeledMetricsOutputDirectory"]
    dynamic_metrics_subdir = config["OUTPUT"]["DynamicMetricsOutputDirectory"]
    output_metrics_subdir = config["OUTPUT"]["AllMetricsOutputDirectory"]

    if config['DYNAMIC'].get('SkipMerge', 'No').lower() == 'yes':
        print("Merging has already been done. Skipping...")
        return

    static_dir = join(data_directory, static_metrics_subdir)
    dynamic_dir = join(data_directory, dynamic_metrics_subdir)
    output_dir = join(data_directory, output_metrics_subdir)

    if not exists(output_dir):
        makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Verify that the directories exist ------------------------------------ #
    if not exists(static_dir):
        raise FileNotFoundError(f"Static metrics directory not found: {static_dir}")
    if not exists(dynamic_dir):
        raise FileNotFoundError(f"Dynamic metrics directory not found: {dynamic_dir}")

    # Merge each pair of static and dynamic CSVs --------------------------- #
    for dynamic_file in os.listdir(dynamic_dir):
        if dynamic_file.endswith("_dynamic_metrics.csv"):
            version = dynamic_file.replace("_dynamic_metrics.csv", "")
            static_file = join(static_dir, f"{version}_labeled_metrics.csv")
            dynamic_file_path = join(dynamic_dir, dynamic_file)
            output_file = join(output_dir, f"{version}_all_metrics.csv")

            # Check if the corresponding static CSV exists
            if not exists(static_file):
                print(f"Static metrics file missing for version {version}. Skipping...")
                continue

            # Load the static and dynamic CSVs
            static_df = pd.read_csv(static_file)
            dynamic_df = pd.read_csv(dynamic_file_path)

            # Merge the CSVs on the 'Name' column
            merged_df = pd.merge(static_df, dynamic_df, on='Name', how='outer')

            # Fill missing values with 0
            merged_df.fillna(0, inplace=True)

            # Save the merged CSV
            merged_df.to_csv(output_file, index=False)
            print(f"Merged metrics CSV created: {output_file}")
