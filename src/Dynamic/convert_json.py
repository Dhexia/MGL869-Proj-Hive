import json

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