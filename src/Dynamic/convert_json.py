import json
import pandas as pd
import os

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


def convert_json_to_csv(json_file, output_dir):
    """
    Convert a JSON of dynamic metrics to multiple CSV files, one per version.

    Args:
        json_file (str): Path to the JSON file containing dynamic metrics.
        output_dir (str): Directory to save the output CSV files.
    """
    # Charger le fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir chaque version
    for version, metrics in data.items():
        # Construire les données pour le DataFrame
        rows = []
        for file_name in metrics['count_lines']:
            rows.append({
                'Name': file_name,
                'LinesAdded': metrics['count_lines'][file_name].get('LinesAdded', 0),
                'LinesDeleted': metrics['count_lines'][file_name].get('LinesDeleted', 0),
                'CommitCount': metrics['commit_count'].get(file_name, 0),
                'DeveloperCount': metrics['developer_count'].get(file_name, 0),
            })

        # Créer un DataFrame et trier par nom de fichier
        df = pd.DataFrame(rows).sort_values(by='Name')

        # Sauvegarder en CSV
        output_path = os.path.join(output_dir, f"{version}_dynamic_metrics.csv")
        df.to_csv(output_path, index=False)
        print(f"CSV écrit : {output_path}")
