import os
import pandas as pd
from configparser import ConfigParser

def merge_static_metrics():
    """
    Merge all static metrics CSV files into a single file and add a `Version` column at the first position.
    
    The CSV files are located in the directory specified by `static_metrics_output`
    in the `config.ini` file. The `Version` column is extracted from the file names.
    
    The final merged file is saved in the directory specified by `FullStaticMetricsOutputDirectory`
    in the `config.ini` file.
    """
    # Load the configuration file
    config = ConfigParser()
    config.read("config.ini")
    
    # Retrieve parameters from the configuration
    base_dir = config["GENERAL"]["DataDirectory"]
    static_metrics_dir = config["OUTPUT"]["StaticMetricsOutputDirectory"]
    csv_separator = config["GENERAL"]["CSVSeparatorMetrics"]

    if config['UNDERSTAND'].get('SkipMerge', 'No').lower() == 'yes':
        print("Merging has already been done. Skipping...")
        return
    
    # Construct the full path to the directory containing the CSV files
    metrics_dir = os.path.join(base_dir, static_metrics_dir)
    
    if not os.path.exists(metrics_dir):
        raise FileNotFoundError(f"The directory {metrics_dir} does not exist.")
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(metrics_dir) if f.endswith(".csv")]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in the directory {metrics_dir}.")
    
    merged_data = []  # List to store DataFrames
    
    for csv_file in csv_files:
        # Extract the version from the file name (e.g., `2.1.0` from `2.1.0_labeled_static_metrics.csv`)
        version = os.path.splitext(csv_file)[0].split("_")[0]
        
        # Load the CSV file
        file_path = os.path.join(metrics_dir, csv_file)
        df = pd.read_csv(file_path, sep=csv_separator)
        
        # Add the `Version` column
        df.insert(0, "Version", version)  # Insert at the first position
        
        # Append the DataFrame to the list
        merged_data.append(df)
    
    # Concatenate all DataFrames
    merged_df = pd.concat(merged_data, ignore_index=True)

    # Sort the merged DataFrame by the `Name` column
    if "Name" in merged_df.columns and "Version" in merged_df.columns:
        merged_df = merged_df.sort_values(by=["Name", "Version"])
    
    # Define the output path for the merged file
    output_dir = os.path.join(base_dir, config["UNDERSTAND"]["FullStaticMetricsOutputDirectory"])
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    output_file = os.path.join(output_dir, "merged_static_metrics.csv")
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False, sep=csv_separator)
    
    print(f"Merged file saved at: {output_file}")
