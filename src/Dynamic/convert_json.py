import json
import pandas as pd
from os.path import exists, join
from configparser import ConfigParser
from os import makedirs

config = ConfigParser()
config.read("config.ini")

def save_dynamic_metrics_to_json(dynamic_metrics, file_path):
    """
    Save the dynamic metrics object to a JSON file.

    Args:
        dynamic_metrics (dict): The dynamic metrics object to save.
        file_path (str): Path to the output JSON file.

    Returns:
        None
    """
    try:
        # Write the dynamic metrics to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(dynamic_metrics, json_file, indent=4)
        print(f"Dynamic metrics successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving dynamic metrics: {e}")

def convert_json_to_csv():
    """
    Convert dynamic metrics JSON to CSV files for each version, using configuration from config.ini.

    The function reads:
        - The path of the dynamic metrics JSON from the configuration file.
        - The output directory for saving the generated CSV files.
    
    Returns:
        None
    """

    # Load configuration from config.ini ----------------------------------- #
    dynamic_metrics_file = config["DYNAMIC"]["MetricsFile"]
    data_directory = config["GENERAL"]["DataDirectory"]
    dynamic_metrics_subdir = config["DYNAMIC"]["JsonMetricsSubDirectory"]
    csv_output_subdir = config["OUTPUT"]["DynamicMetricsOutputDirectory"]

    json_path = join(data_directory, dynamic_metrics_subdir, dynamic_metrics_file)
    csv_output_dir = join(data_directory, csv_output_subdir)

    if not exists(csv_output_dir):
        makedirs(csv_output_dir)
        print(f"Created output directory: {csv_output_dir}")

    if config['DYNAMIC'].get('SkipConversion', 'No').lower() == 'yes':
        print("Conversion of dynamic metrics to csv has already been done. Skipping...")
        return
    
    # Load the dynamic metrics JSON ---------------------------------------- #
    if not exists(json_path):
        raise FileNotFoundError(f"Dynamic metrics JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        dynamic_metrics = json.load(f)

    # Convert JSON to CSV for each version --------------------------------- #
    for version, metrics in dynamic_metrics.items():
        rows = []
        for file_name in metrics['count_lines']:
            rows.append({
                'Name': file_name,
                'LinesAdded': metrics['count_lines'][file_name].get('LinesAdded', 0),
                'LinesDeleted': metrics['count_lines'][file_name].get('LinesDeleted', 0),
                'CommitCount': metrics['commit_count'].get(file_name, 0),
                'DeveloperCount': metrics['developer_count'].get(file_name, 0),
                'BugFixCount': metrics['bug_fix_count'].get(file_name, 0),
                'CumulativeDeveloperCount': metrics['cumulative_developer_count'].get(file_name, 0),
                'AverageTimeBetweenChanges': metrics['average_time_between_changes'].get(file_name, 0),
                'CumulativeCommitCount': metrics['cumulative_commit_count'].get(file_name, 0),
                'ChangedComments': metrics['comment_change_metrics'][file_name].get('ChangedComments', 0),
                'UnchangedComments': metrics['comment_change_metrics'][file_name].get('UnchangedComments', 0),
                'CumulativeChangedComments': metrics['cumulative_comment_metrics'][file_name].get('ChangedComments', 0),
                'CumulativeUnchangedComments': metrics['cumulative_comment_metrics'][file_name].get('UnchangedComments', 0),
                'CumulativeAverageTimeBetweenChanges': metrics['cumulative_average_time_between_changes'].get(file_name, 0),
                'AverageExpertise': metrics['average_expertise'].get(file_name, 0),
                'MinimumExpertise': metrics['minimum_expertise'].get(file_name, 0)
            })

        # Create a DataFrame and sort by file name
        df = pd.DataFrame(rows).sort_values(by='Name')

        # Save the CSV file
        csv_file_path = join(csv_output_dir, f"{version}_dynamic_metrics.csv")
        df.to_csv(csv_file_path, index=False)
        print(f"Dynamic metrics CSV created: {csv_file_path}")
