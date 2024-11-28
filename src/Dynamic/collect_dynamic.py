import time
from packaging.version import Version
from Dynamic.collect_sub_dynamic import collect_count_lines, collect_commit_count, collect_developer_count
from Dynamic.convert_json import save_dynamic_metrics_to_json
from configparser import ConfigParser
from os import path, cpu_count, makedirs
from os.path import exists, join
from concurrent.futures import ProcessPoolExecutor

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

    # Load configuration ---------------------------------------------------- #

    file_path, threads_num, main_repo_path = load_configuration()
    if skip_dynamic_metrics():
        return

    # Initialize variables -------------------------------------------------- #

    serialize = threads_num > 1  # False if sequential mode enabled
    version_metrics, start_index, end_index, total_pairs, versions = initialize_versions(
        sorted_versions, start_version, limit, serialize=serialize
    )

    # Retrieve the dynamic metrics ------------------------------------------ #

    if threads_num == 1:
        version_metrics = run_sequential_mode(start_index, end_index, versions, repo, version_metrics, total_pairs)
    else:
        version_metrics = run_parallel_mode(start_index, end_index, versions, version_metrics, main_repo_path, threads_num)

    # Save the dynamic metrics to a JSON file -------------------------------- #

    save_dynamic_metrics_to_json(version_metrics, file_path)

    return version_metrics

def run_sequential_mode(start_index, end_index, versions, repo, version_metrics, total_pairs):
    """
    Run the dynamic metrics collection in sequential mode.

    Args:
        start_index (int): Starting index for processing.
        end_index (int): Ending index for processing.
        versions (list): List of version-commit pairs.
        repo (Repo): The main Git repository.
        version_metrics (dict): Dictionary to store collected metrics.
        total_pairs (int): Total pairs of versions to process.

    Returns:
        dict: Collected dynamic metrics.
    """
    print("Starting sequential processing...")

    for i in range(start_index, min(end_index, len(versions) - 1)):
        version_metrics = process_version_pair(
            i, total_pairs, versions=versions, repo=repo, metrics=version_metrics
        )
    return version_metrics

def run_parallel_mode(start_index, end_index, versions, version_metrics, main_repo_path, max_cores):
    """
    Run the dynamic metrics collection in parallel mode.

    Args:
        start_index (int): Starting index for processing.
        end_index (int): Ending index for processing.
        versions (list): List of version-commit pairs.
        version_metrics (dict): Dictionary to store collected metrics.
        main_repo_path (str): Path to the main Git repository.
        max_cores (int): Maximum number of cores to use for processing.

    Returns:
        dict: Collected dynamic metrics.
    """

    print(f"Starting parallel processing with {max_cores} cores...")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = [
            executor.submit(process_version_pair_safe, i, versions, main_repo_path, version_metrics)
            for i in range(start_index, min(end_index, len(versions) - 1))
        ]
        for future in futures:
            version_metrics.update(future.result())  # Merge results from each process

    
    elapsed_time = time.time() - start_time

    print(f"Time spent: {elapsed_time:.2f} seconds")

    return version_metrics

def process_version_pair(i, total_pairs, versions, repo, metrics):
    """
    Process a pair of versions and collect metrics for the commits between them.

    Args:
        i (int): Current index in the versions list.
        total_pairs (int): Total pairs to process.
        versions (list): List of version-commit pairs (from initialize_versions).
        repo (Repo): The main Git repository.
        metrics (dict): Collected metrics.

    Returns:
        dict: Updated metrics after processing the current pair.
    """
    # Access the current and next version using the list format
    current_version, current_commit = versions[i]
    next_version, next_commit = versions[i + 1]

    print(f"Processing commits between {current_version} ({current_commit.hexsha}) "
          f"and {next_version} ({next_commit.hexsha})")

    # Start timing this pair
    start_time = time.time()

    # Get the commit range between current and next version
    commit_range = list(repo.iter_commits(f"{current_commit.hexsha}..{next_commit.hexsha}"))

    # Collect metrics for this range
    metrics[next_version] = collect_metrics_for_commits(commit_range)

    # Calculate time taken
    elapsed_time = time.time() - start_time

    # Show progress and time
    progress = ((i+1) / total_pairs) * 100

    print(f"Progress: {progress:.2f}% ({i+1}/{total_pairs}) | Time spent: {elapsed_time:.2f} seconds")

    return metrics

def process_version_pair_safe(i, versions, repo_path, metrics):
    """
    Safe version of process_version_pair to be executed in a separate process.

    Args:
        i (int): Current index in the versions list.
        versions (list): List of version-commit pairs.
        repo_path (str): Path to the Git repository.
        metrics (dict): Collected metrics.

    Returns:
        dict: Updated metrics for the processed pair.
    """
    from git import Repo
    repo = Repo(repo_path)

    current_version, current_commit_hexsha = versions[i]
    next_version, next_commit_hexsha = versions[i + 1]  

    # Recreate Commit objects
    current_commit = repo.commit(current_commit_hexsha)
    next_commit = repo.commit(next_commit_hexsha)

    print(f"Processing commits between {current_version} ({current_commit.hexsha}) "
      f"and {next_version} ({next_commit.hexsha})", flush=True)

    commit_range = list(repo.iter_commits(f"{current_commit.hexsha}..{next_commit.hexsha}"))
    metrics[next_version] = collect_metrics_for_commits(commit_range)
    return metrics

def initialize_versions(sorted_versions, start_version, limit, serialize=False):
    """
    Initialize version indices and limits for processing.

    Args:
        sorted_versions (dict): Dictionary with versions and commits.
        start_version (str): The version to start from.
        limit (int): Maximum number of pairs to process.

    Returns:
        tuple: Initialized metrics, start index, end index, total pairs.
    """
    version_metrics = {}

    if serialize:
        versions = [(version, commit.hexsha) for version, commit in sorted_versions.items()]
    else:
        versions = list(sorted_versions.items())

    start_index = 0
    if start_version:
        if start_version in sorted_versions:
            start_index = list(sorted_versions.keys()).index(start_version)
        else:
            raise ValueError(f"Start version {start_version} not found in sorted_versions.")

    # Apply limit if specified
    end_index = start_index + limit if limit is not None else len(versions)
    total_pairs = min(end_index, len(versions) - 1) - start_index
    return version_metrics, start_index, end_index, total_pairs, versions

def collect_metrics_for_commits(commits):
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

def load_configuration():
    """
    Load configuration from config.ini.

    Returns:
        tuple: Metrics file name, data directory, and full file path.
    """

    # Paths and directories
    metrics_file = config["DYNAMIC"]["MetricsFile"]
    data_directory = config["GENERAL"]["DataDirectory"]
    json_subdir = config["DYNAMIC"]["JsonMetricsSubDirectory"]
    json_metrics_path = join(data_directory, json_subdir)

    if not exists(json_metrics_path):
        makedirs(json_metrics_path)
        print(f"Created directory: {json_metrics_path}")

    file_path = join(json_metrics_path, metrics_file)

    # CPU
    hive_git_directory: str = config["GIT"]["HiveGitDirectory"]
    hive_git_repo_name: str = config["GIT"]["HiveGitRepoName"]
    data_directory: str = config["GENERAL"]["DataDirectory"]
    main_repo_path: str = path.abspath(path.join(data_directory, hive_git_directory, hive_git_repo_name))   
    
    max_threads: int = int(config["GENERAL"]["MaxThreads"])
    threads_num: int = min(max_threads, cpu_count())

    return file_path, threads_num, main_repo_path

def skip_dynamic_metrics():
    """
    Check if dynamic metrics collection should be skipped based on the configuration.

    Returns:
        bool: True if skipping is enabled, False otherwise.
    """
    if config['DYNAMIC'].get('SkipDynamicAnalysis', 'No').lower() == 'yes':
        print("Dynamic Metrics have already been collected. Skipping...")
        return True
    return False
