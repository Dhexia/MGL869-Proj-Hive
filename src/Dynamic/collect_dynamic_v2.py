from configparser import ConfigParser
from Dynamic.convert_json import save_dynamic_metrics_to_json
from concurrent.futures import ThreadPoolExecutor
import time
from threading import Lock
from Dynamic.collect_sub_dynamic import collect_count_lines, collect_commit_count, collect_developer_count, collect_bug_fix_count, collect_cumulative_developer_count, collect_average_time_between_changes, collect_cumulative_commit_count, collect_comment_change_metrics, collect_cumulative_comment_metrics, collect_cumulative_average_time_between_changes, collect_average_expertise, collect_minimum_expertise
from os.path import exists, join, abspath
from os import cpu_count, makedirs

print_lock = Lock()
config=ConfigParser()
config.read('config.ini')

def collect_dynamic_metrics_v2(version_json, limit=None):
    """
    Collect dynamic metrics for each version starting from a configured start_version,
    based on commits strictly between a version and its parent, handling branching.

    Args:
        version_json (dict): Dictionary with version metadata.
        limit (int, optional): Maximum number of versions to process.

    Returns:
        dict: Collected dynamic metrics.
    """
    # Load configuration ---------------------------------------------------- #
    config = ConfigParser()
    config.read("config.ini")

    start_version = config["DYNAMIC"]["StartV"]
    file_path, threads_num, main_repo_path = load_configuration()
    if skip_dynamic_metrics():
        return

    # Initialize variables -------------------------------------------------- #
    version_metrics = {}
    versions_to_process = [start_version]

    # Choose between sequential and parallel processing --------------------- #
    if threads_num > 1:
        version_metrics = run_parallel_mode_with_threads(
            version_json, versions_to_process, version_metrics, main_repo_path, threads_num, limit
        )
    else:
        version_metrics = run_sequential_mode_v2(
            version_json, versions_to_process, version_metrics, main_repo_path, limit
        )

    # Save the dynamic metrics to a JSON file ------------------------------- #
    save_dynamic_metrics_to_json(version_metrics, file_path)

    return version_metrics


def process_version_pair_safe_v2(current_version, version_json, repo_path, metrics):
    """
    Safe version of process_version_pair to be executed in a separate process, adapted for version_json.

    Args:
        current_version (str): Current version to process.
        version_json (dict): Dictionary with version metadata.
        repo_path (str): Path to the Git repository.
        metrics (dict): Collected metrics.

    Returns:
        dict: Updated metrics for the processed pair.
    """
    from git import Repo
    repo = Repo(repo_path)

    # Get the previous version
    previous_version = version_json[current_version]['previous']

    # If there's no previous version, skip processing
    if not previous_version:
        return metrics

    # Get the commit objects
    current_commit = repo.commit(version_json[current_version]['commit'].hexsha)
    previous_commit = repo.commit(version_json[previous_version]['commit'].hexsha)

    with print_lock:
        print(f"Processing commits between {previous_version} ({previous_commit.hexsha}) "
            f"and {current_version} ({current_commit.hexsha})", flush=True)

    # Get the range of commits between the previous and current version
    commit_range = list(repo.iter_commits(f"{previous_commit.hexsha}..{current_commit.hexsha}"))
    metrics[current_version] = collect_metrics_for_commits(commit_range)

    return metrics


def run_parallel_mode_with_threads(version_json, versions_to_process, version_metrics, main_repo_path, max_threads, limit=None):
    """
    Run the dynamic metrics collection in parallel mode, handling branches.

    Args:
        version_json (dict): Dictionary with version metadata.
        versions_to_process (list): List of versions to process.
        version_metrics (dict): Dictionary to store collected metrics.
        main_repo_path (str): Path to the main Git repository.
        max_threads (int): Maximum number of threads to use for processing.
        limit (int, optional): Maximum number of versions to process.

    Returns:
        dict: Collected dynamic metrics.
    """
    print(f"Starting parallel processing with {max_threads} threads...")

    start_time = time.time()
    processed_versions = set()
    futures = []
    iteration_count = 0

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        while versions_to_process:
            if limit is not None and iteration_count >= limit:
                print(f"Limit of {limit} versions reached. Stopping early.", flush=True)
                break

            current_version = versions_to_process.pop(0)
            if current_version in processed_versions:
                continue  # Skip already processed versions

            # Schedule the current version for processing
            futures.append(executor.submit(
                process_version_pair_safe_v2, current_version, version_json, main_repo_path, version_metrics
            ))

            # Add all `next` versions to the queue
            next_versions = version_json[current_version]['next']
            if next_versions:
                versions_to_process.extend(next_versions)

            processed_versions.add(current_version)
            iteration_count += 1

        # Collect results from all threads
        for future in futures:
            version_metrics.update(future.result())  # Merge results from each thread

    elapsed_time = time.time() - start_time
    print(f"Time spent: {elapsed_time:.2f} seconds")

    return version_metrics


def run_sequential_mode_v2(version_json, versions_to_process, version_metrics, main_repo_path, limit=None):
    """
    Run the dynamic metrics collection in sequential mode, handling branching.

    Args:
        version_json (dict): Dictionary with version metadata.
        versions_to_process (list): List of versions to process.
        version_metrics (dict): Dictionary to store collected metrics.
        main_repo_path (str): Path to the main Git repository.
        limit (int, optional): Maximum number of versions to process. Defaults to None (no limit).

    Returns:
        dict: Collected dynamic metrics.
    """
    print("Starting sequential processing...")

    processed_versions = set()  # To track already processed versions
    iteration_count = 0

    while versions_to_process:
        if limit is not None and iteration_count >= limit:
            print(f"Limit of {limit} versions reached. Stopping early.", flush=True)
            break

        current_version = versions_to_process.pop(0)
        if current_version in processed_versions:
            continue  # Skip already processed versions

        # Process the current version and its previous version
        version_metrics = process_version_pair_safe_v2(current_version, version_json, main_repo_path, version_metrics)

        # Add all `next` versions to the queue
        next_versions = version_json[current_version]['next']
        if next_versions:
            versions_to_process.extend(next_versions)

        processed_versions.add(current_version)
        iteration_count += 1

    return version_metrics


def load_configuration():
    """
    Load configuration from config.ini.

    Returns:
        tuple: Metrics file name, data directory, and full file path.
    """

    # Paths and directories
    metrics_file = config["DYNAMIC"]["MetricsFile"]
    data_directory = config["GENERAL"]["DataDirectory"]
    json_subdir = config["DYNAMIC"]["AllMetricsSubDir"]
    json_metrics_path = join(data_directory, json_subdir)

    if not exists(json_metrics_path):
        makedirs(json_metrics_path)
        print(f"Created directory: {json_metrics_path}")

    file_path = join(json_metrics_path, metrics_file)

    # CPU
    hive_git_directory: str = config["GIT"]["HiveGitDirectory"]
    hive_git_repo_name: str = config["GIT"]["HiveGitRepoName"]
    main_repo_path: str = abspath(join(data_directory, hive_git_directory, hive_git_repo_name))   
    
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
        "bug_fix_count": dict(collect_bug_fix_count(commits)),
        "cumulative_developer_count": dict(collect_cumulative_developer_count(commits)),
        "average_time_between_changes": dict(collect_average_time_between_changes(commits)),
        "cumulative_commit_count": dict(collect_cumulative_commit_count(commits)),
        "comment_change_metrics": dict(collect_comment_change_metrics(commits)),
        "cumulative_comment_metrics": dict(collect_cumulative_comment_metrics(commits)),
        "cumulative_average_time_between_changes": dict(collect_cumulative_average_time_between_changes(commits)),
        "average_expertise": dict(collect_average_expertise(commits)),
        "minimum_expertise": dict(collect_minimum_expertise(commits))
    }
    return metrics
