import pandas as pd
from configparser import ConfigParser
from os import path, remove

# Load global configuration
config = ConfigParser()
config.read("config.ini")

base_dir = config["GENERAL"]["DataDirectory"]
output_dir = path.join(base_dir, config["OUTPUT"]["StaticMetricsOutputDirectory"])
csv_separator = config["GENERAL"].get("CSVSeparatorMetrics", ",")

def merge_all_metrics(versions):
    """
    Merges enriched and labeled metrics for a list of versions.

    Args:
        versions (list): List of versions to process (e.g., ["2.0.0", "2.0.1", ...]).

    The function:
    - Merges labeled and enriched metrics files based on "Name" (labeled) and "FileName" (enriched).
    - Completes missing values in labeled metrics with enriched metrics.
    - Replaces remaining NaN values with 0.
    - Removes the 'Kind' column from the final metrics file.
    - Saves the final metrics to a new CSV file.
    - Deletes the original labeled and enriched metrics files.
    """
    if config['UNDERSTAND'].get('SkipMerge', 'No').lower() == 'yes':
        print("Merging has already been done. Skipping...")
        return
    
    for version in versions:
        try:
            # File paths
            labeled_file = path.join(output_dir, f"{version}_labeled_metrics.csv")
            enrichi_file = path.join(output_dir, f"{version}_enrichi_metrics.csv")
            final_file = path.join(output_dir, f"{version}_static_metrics.csv")
            
            # Load the files
            labeled_metrics = pd.read_csv(labeled_file, sep=csv_separator)
            enrichi_metrics = pd.read_csv(enrichi_file, sep=csv_separator)
            
            # Identify common columns (excluding 'Name' and 'FileName')
            common_columns = [
                col for col in labeled_metrics.columns if col in enrichi_metrics.columns
            ]
            
            # Merge data on "Name" (labeled) and "FileName" (enriched)
            merged_metrics = pd.merge(
                labeled_metrics,
                enrichi_metrics,
                how="left",
                left_on="Name",
                right_on="FileName"
            )
            
            # Complete missing values in labeled metrics with enriched metrics
            for column in common_columns:
                merged_metrics[column] = merged_metrics[f"{column}_x"].combine_first(
                    merged_metrics[f"{column}_y"]
                )
            
            # Keep only the original columns from labeled metrics
            final_metrics = merged_metrics[labeled_metrics.columns]
            
            # Replace remaining NaN values with 0
            final_metrics = final_metrics.fillna(0)
            
            # Remove the 'Kind' column if it exists
            if "Kind" in final_metrics.columns:
                final_metrics = final_metrics.drop(columns=["Kind"])
            
            # Save the final metrics file
            final_metrics.to_csv(final_file, sep=csv_separator, index=False)
            
            # Delete the original source files
            remove(labeled_file)
            remove(enrichi_file)
            
            print(f"Final file generated for version {version}: {final_file}")
        
        except Exception as e:
            print(f"Error processing version {version}: {e}")
