from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from re import Pattern, compile
from os import path, cpu_count
from git import Repo, Commit


# Function to process a batch of commits
def process_commits(ids: set, commits: [Commit], repo_dir: str):
    # Load the repository in memory of the current thread
    local_repo: Repo = Repo(repo_dir)

    tuple_key_file_commit = []
    for commit_id in commits:
        for match in commits[commit_id]:
            hive_key = f'HIVE-{match}'
            if hive_key in ids:
                for file in local_repo.commit(commit_id).stats.files:
                    tuple_key_file_commit.append((hive_key, file, commit_id))
    return tuple_key_file_commit


def commit_analysis(ids: set) -> [(str, str, str)]:
    """
    Analyze the commits in the Hive repository.
    Repository need to be cloned before calling this function.
    :return:
    """

    config = ConfigParser()
    config.read('config.ini')

    # Read the config file -------------------------------------------------- #

    hive_git_directory: str = config["GIT"]["HiveGitDirectory"]
    hive_git_repo_name: str = config["GIT"]["HiveGitRepoName"]
    commit_pattern: Pattern = compile(config["GIT"]["CommitPattern"])
    max_threads: int = int(config["GENERAL"]["MaxThreads"])
    data_directory: str = config["GENERAL"]["DataDirectory"]

    # Variables ------------------------------------------------------------- #

    # Get the number of threads
    num_threads: int = min(max_threads, cpu_count())

    # List to store the couples (issue, file, commit)
    all_couples: [(str, str, str)] = []
    # Repo path
    repo_path: str = path.join(data_directory, hive_git_directory, hive_git_repo_name)
    # Load the repository
    repo: Repo = Repo(repo_path)
    # Split the commits into chunks
    chunk_size: int = len(list(repo.iter_commits())) // num_threads
    # Get all commits and files
    all_commits: [dict] = [{} for _ in range(num_threads)]

    # Prepare multi-threading chunks of commits -------------------------------- #

    for i, commit in enumerate(repo.iter_commits()):
        matches = commit_pattern.findall(commit.message)
        if matches:
            all_commits[i // chunk_size][commit.hexsha] = matches

    # Process the commits in parallel ------------------------------------------ #

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_commits, ids, chunk, repo_path) for chunk in all_commits]
        for future in as_completed(futures):
            couples = future.result()
            all_couples.extend(couples)

    print(f"{len(all_couples)} couples found.")

    return all_couples
