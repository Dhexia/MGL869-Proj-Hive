import time
from packaging.version import Version
from Dynamic.collect_sub_dynamic import collect_count_lines, collect_commit_count, collect_developer_count
from Dynamic.convert_json import save_dynamic_metrics_to_json
from configparser import ConfigParser
from os import path

# Configuration loaded once globally
config = ConfigParser()
config.read("config.ini")

def collect_dynamic_metrics(sorted_versions, repo, start_version=None, limit=None):
    """
    Collect dynamic metrics for each version based on commits strictly between two consecutive tags.

    Args:
        sorted_versions (dict): Dictionary with versions as keys and commits as values.
        repo (Repo): The main Git repository.
        start_version (str, optional): The version from which to start collecting metrics.
        limit (int, optional): Maximum number of version pairs to analyze. Defaults to None (no limit).

    Returns:
        dict: Collected dynamic metrics for the specified versions.
    """

    metrics_file : str = config["DYNAMIC"]["MetricsFile"]
    data_directory: str = config["GENERAL"]["DataDirectory"]
    file_path = path.join(data_directory, metrics_file)

    if config['DYNAMIC'].get('SkipDynamic', 'No').lower() == 'yes':
        print("Dynamic Metrics have already been collected. Skipping...")
        return

    version_metrics = {}
    versions = list(sorted_versions.items())  # Convert to list for indexing

    # Find the starting index
    start_index = 0
    if start_version:
        if start_version in sorted_versions:
            start_index = list(sorted_versions.keys()).index(start_version)
        else:
            raise ValueError(f"Start version {start_version} not found in sorted_versions.")

    # Apply limit if specified
    end_index = start_index + limit if limit is not None else len(versions)

    total_pairs = min(end_index, len(versions) - 1) - start_index  # Total pairs to process

    for i in range(start_index, min(end_index, len(versions) - 1)):  # Iterate over pairs of versions
        current_version, current_commit = versions[i]
        next_version, next_commit = versions[i + 1]

        print(f"Processing commits between {current_version} ({current_commit.hexsha}) "
              f"and {next_version} ({next_commit.hexsha})")

        # Start timing this pair
        start_time = time.time()

        # Get the commit range between current and next version
        commit_range = list(repo.iter_commits(f"{current_commit.hexsha}..{next_commit.hexsha}"))

        # Collect metrics for this range
        version_metrics[next_version] = collect_metrics_for_commits(commit_range, repo)

        # Calculate time taken
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Show progress and time
        progress = (i / total_pairs) * 100
        print(f"Progress: {progress:.2f}% ({i}/{total_pairs}) | Time spent: {elapsed_time:.2f} seconds")

    # Save the dynamic metrics to a JSON file
    save_dynamic_metrics_to_json(version_metrics, file_path)

    return version_metrics




def collect_metrics_for_commits(commits, repo):
    """
    Collect metrics for a range of commits.

    Args:
        commits (list): List of commits for the current version.
        repo (Repo): The main Git repository.

    Returns:
        dict: Collected metrics for the commits.
    """
    metrics = {
        "count_lines": dict(collect_count_lines(commits)),  # Convert defaultdict to dict for JSON serialization
        "commit_count": dict(collect_commit_count(commits)),
        "developer_count": dict(collect_developer_count(commits)),
    }
    return metrics
