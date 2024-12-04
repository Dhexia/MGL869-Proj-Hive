import pandas as pd
from os import path, listdir, makedirs
from configparser import ConfigParser

# Configuration loaded once globally
config = ConfigParser()
config.read("config.ini")

def enrich_metrics(couples_df: pd.DataFrame) -> None:
    """
    Extracts metrics of all kinds except "File" and those containing "Function" in the Kind field
    and saves the results in an enrichment output directory.

    Parameters:
        couples_df (pd.DataFrame): DataFrame containing files with issues and their affected versions.
    """
    base_dir = config["GENERAL"]["DataDirectory"]
    metrics_directory = path.join(base_dir, config["OUTPUT"]["MetricsOutputDirectory"])
    output_dir = path.join(base_dir, config["OUTPUT"]["StaticMetricsOutputDirectory"])
    csv_separator = config["GENERAL"].get("CSVSeparatorMetrics", ",")

    if config['UNDERSTAND'].get('SkipEnrich', 'No').lower() == 'yes':
        print("Enrichment process is skipped as per configuration.")
        return

    ensure_directory_exists(output_dir)

    if not path.exists(metrics_directory):
        raise FileNotFoundError(f"Metrics directory not found: {metrics_directory}")

    for metrics_file in listdir(metrics_directory):
        if metrics_file.endswith(".csv"):
            process_enrichment_metrics(metrics_file, metrics_directory, output_dir, csv_separator)

def ensure_directory_exists(directory: str) -> None:
    """Ensures the specified directory exists, creating it if necessary."""
    if not path.exists(directory):
        print(f"Creating output directory: {directory}")
        makedirs(directory)

def extract_java_filename(name: str) -> str:
    """
    Extracts the corresponding Java file name from a fully qualified name.
    
    Parameters:
        name (str): The fully qualified name (e.g., 'org.apache.hadoop.fs.DefaultFileAccess.getSuperGroupName').
    
    Returns:
        str: The Java file name (e.g., 'DefaultFileAccess.java').
    """
    # Split the name by '.'
    parts = name.split('.')
    
    # Look for the last part that starts with an uppercase letter
    for part in reversed(parts):
        if part and part[0].isupper():  # Ensure it's not empty and starts with an uppercase letter
            return f"{part}.java"
    
    # Return a default value if no match is found
    return ""

def process_enrichment_metrics(metrics_file: str, metrics_directory: str, 
                               output_dir: str, csv_separator: str) -> None:
    """
    Processes a single metrics file, aggregates metrics by FileName,
    and saves the enriched output.
    """
    metrics_path = path.join(metrics_directory, metrics_file)
    print(f"Processing enrichment metrics file: {metrics_path}")

    try:
        metrics_df = load_metrics_file(metrics_path, csv_separator)

        filtered_df = filter_metrics(metrics_df)

        if filtered_df.empty:
            print(f"No enrichment metrics found in {metrics_file}.")
            return

        enriched_df = prepare_enriched_metrics(filtered_df)

        # Aggregate metrics by FileName
        aggregated_df = aggregate_metrics_by_file(enriched_df)

        version = extract_version(metrics_file)

        save_enriched_metrics(aggregated_df, output_dir, version, csv_separator)
    except Exception as e:
        print(f"Error processing enrichment metrics file {metrics_file}: {e}")


def load_metrics_file(metrics_path: str, csv_separator: str) -> pd.DataFrame:
    """Loads the metrics file and ensures required columns exist."""
    metrics_df = pd.read_csv(metrics_path, sep=csv_separator, engine="python")
    required_columns = ["Kind", "Name"]
    if not all(col in metrics_df.columns for col in required_columns):
        raise KeyError(f"Required columns {required_columns} not found in {metrics_path}")
    return metrics_df


def filter_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Applies filtering conditions on the metrics DataFrame."""
    return metrics_df[
        ((metrics_df["Kind"].str.contains("Method", na=False, case=False) |
          (metrics_df["Kind"].str.contains("Class", na=False, case=False) &
           (metrics_df["Kind"] != "Class"))) &
         (~metrics_df["Kind"].str.contains("File", na=False, case=False)) &
         (~metrics_df["Kind"].str.contains("Function", na=False, case=False)) &
         (metrics_df["Name"].str.startswith("org.", na=False)))
    ]


def prepare_enriched_metrics(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps specific columns, adds 'FileName', and prepares the DataFrame
    for saving.
    """
    columns_to_keep = [
        "CountClassBase", "CountClassCoupled", "CountClassCoupledModified", "CountClassDerived",
        "CountDeclFile", "CountDeclMethodAll", "CountInput", "CountOutput", "Cyclomatic",
        "MaxInheritanceTree", "PercentLackOfCohesion", "PercentLackOfCohesionModified",
        "CountDeclFileCode", "CountDeclFileHeader", "CountDeclInstanceVariablePrivate",
        "CountDeclInstanceVariableProtected", "CountDeclInstanceVariablePublic",
        "CountDeclMethodConst", "CountDeclMethodFriend", "CountLineInactive",
        "CountLinePreprocessor", "CountStmtEmpty"
    ]

    columns_to_keep_with_meta = ["Kind", "Name"] + columns_to_keep
    enriched_df = filtered_df[columns_to_keep_with_meta]
    enriched_df.insert(2, "FileName", enriched_df["Name"].apply(extract_java_filename))
    return enriched_df


def extract_version(metrics_file: str) -> str:
    """Extracts the version from the file name."""
    return metrics_file.replace("_metrics.csv", "")


def save_enriched_metrics(enriched_df: pd.DataFrame, output_dir: str, version: str, csv_separator: str) -> None:
    """Saves the enriched metrics DataFrame to the output directory."""
    enriched_metrics_path = path.join(output_dir, f"{version}_enrichi_metrics.csv")
    enriched_df.to_csv(enriched_metrics_path, index=False, sep=csv_separator)
    print(f"Enriched metrics saved to: {enriched_metrics_path}")


def aggregate_metrics_by_file(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics by FileName and updates the Kind column to 'File',
    while removing any empty or invalid rows.
    
    Parameters:
        enriched_df (pd.DataFrame): The DataFrame containing enriched metrics.
    
    Returns:
        pd.DataFrame: Aggregated metrics by FileName with updated Kind.
    """
    # Aggregation rules
    aggregation_rules = {
        "CountClassBase": "sum",
        "CountClassCoupled": "sum",
        "CountClassCoupledModified": "sum",
        "CountClassDerived": "sum",
        "CountDeclFile": "sum",
        "CountDeclMethodAll": "sum",
        "CountInput": "sum",
        "CountOutput": "sum",
        "Cyclomatic": "sum",
        "MaxInheritanceTree": "max",
        "PercentLackOfCohesion": "mean",
        "PercentLackOfCohesionModified": "mean",
        "CountDeclFileCode": "sum",
        "CountDeclFileHeader": "sum",
        "CountDeclInstanceVariablePrivate": "sum",
        "CountDeclInstanceVariableProtected": "sum",
        "CountDeclInstanceVariablePublic": "sum",
        "CountDeclMethodConst": "sum",
        "CountDeclMethodFriend": "sum",
        "CountLineInactive": "sum",
        "CountLinePreprocessor": "sum",
        "CountStmtEmpty": "sum"
    }

    # Perform aggregation by FileName
    aggregated_metrics = enriched_df.groupby("FileName").agg(aggregation_rules).reset_index()

    # Remove rows where FileName is empty or NaN
    aggregated_metrics = aggregated_metrics[aggregated_metrics["FileName"].notna() & (aggregated_metrics["FileName"] != "")]

    # Add the Kind column with value 'File'
    aggregated_metrics["Kind"] = "File"

    # Reorder columns: 'Kind' first, followed by all others
    columns_order = ["Kind", "FileName"] + [col for col in aggregated_metrics.columns if col not in ["Kind", "FileName"]]
    aggregated_metrics = aggregated_metrics[columns_order]

    return aggregated_metrics
