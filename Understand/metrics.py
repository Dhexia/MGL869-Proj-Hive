from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from os import path, cpu_count, makedirs
from git import Repo, Commit
from subprocess import run, CalledProcessError, PIPE
from shutil import rmtree

from Understand.commands import und_create_command, und_add_command, und_analyze_command, und_metrics_command

import glob
import shutil
import os

def create_understand_project(projet_directory: str, name: str):
    """

    :param projet_directory:
    :param name:
    :return:
    """
    projet_path: str = path.join(projet_directory, name)
    und_create_command(project_path=projet_path)


def create_version_repo(commit, version: str) -> (Repo, str):
    """
    Create a temporary repository with the version of the project
    :param version: Version that is being analyzed (tagged on the commit)
    :param commit: The commit object
    :return: The created repository and the path to the Understand project
    """

    # Read the config file -------------------------------------------------- #

    config = ConfigParser()
    config.read('config.ini')

    # Variables ------------------------------------------------------------- #

    hive_git_directory: str = config["GIT"]["HiveGitDirectory"]
    hive_git_repo_name: str = config["GIT"]["HiveGitRepoName"]
    temp_repo_directory: str = config["UNDERSTAND"]["TempRepoDirectory"]
    data_directory: str = config["GENERAL"]["DataDirectory"]

    temp_version_path: str = path.join(data_directory, temp_repo_directory, version)
    main_repo_path: str = path.abspath(path.join(data_directory, hive_git_directory, hive_git_repo_name))

    print(f"Creating repo {temp_version_path} from {main_repo_path}")

    if path.exists(temp_version_path):
        # Delete the directory if it exists
        print(f"Deleting previous the directory: {temp_version_path}")
        rmtree(temp_version_path)

    print(f"Creating the directory: {temp_version_path}")
    makedirs(temp_version_path)

    try:
        # Create the temporary repository ----------------------------------- #

        run(['git', 'init', temp_version_path], check=True, stdout=PIPE, stderr=PIPE, text=True)
        run(['git', '-C', temp_version_path, 'remote', 'add', 'origin', f'file://{main_repo_path}'],
            check=True, stdout=PIPE, stderr=PIPE, text=True)
        run(['git', '-C', temp_version_path, 'fetch', '--depth=1', 'origin', commit.hexsha],
            check=True, stdout=PIPE, stderr=PIPE, text=True)
        run(['git', '-C', temp_version_path, 'restore', '--source', commit.hexsha, '.'],
            check=True, stdout=PIPE, stderr=PIPE, text=True)

        create_understand_project(temp_version_path, version + ".und")

        repo = Repo(temp_version_path)
        return repo, path.join(temp_version_path, version + ".und")

    except CalledProcessError as e:
        print(f"Error occurred while running command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        raise


def analyze_commit(commit: Commit, version: str) -> str:
    """
    Analyze the code for a specific commit
    :param version: Version that is being analyzed (tagged on the commit)
    :param commit: Commit - code source to analyze
    :return: None
    """
    repo, und_project = create_version_repo(commit, version)
    print(f"Analyzing commit {commit.hexsha}")

    # Run the analysis
    und_add_command(repo.working_dir, und_project)

    # Analyze the code
    und_analyze_command(und_project)

    # Extract the metrics
    und_metrics_command(und_project)

    return version


def multi_threaded_analysis(versions: dict, num_threads: int) -> None:
    """
    Analyze code for each commit in the Hive repository
    :param versions: dictionary of commits to analyze
    :param num_threads: Number of threads to use
    :return: None
    """

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit each commit for analysis
        future_to_commit = {executor.submit(analyze_commit, versions[version], version): versions[version] for version
                            in versions}

        i = 1
        # Collect results and handle any exceptions
        for future in as_completed(future_to_commit):
            version = future_to_commit[future]
            try:
                # Retrieve the result (if analyze_commit returns something)
                future.result()
                print(f"{i}/{len(versions)} - Successfully analyzed version: {version}")
            except Exception as e:
                print(f"Error analyzing commit {version}: {e}")
                i += 1


def metrics(versions: dict, limit: int = None):
    """
    :param versions:
    :return:
    """

    config = ConfigParser()
    config.read('config.ini')

    temp_repo_directory: str = config['UNDERSTAND']['TempRepoDirectory']
    data_directory: str = config["GENERAL"]["DataDirectory"]
    metrics_output_directory: str = config["OUTPUT"]["MetricsOutputDirectory"]
    max_threads: int = int(config["GENERAL"]["MaxThreads"])

    if config['UNDERSTAND'].get('SkipMetricsAnalysis', 'No').lower() == 'yes':
        print("Metrics analysis is skipped as per configuration.")
        return

    temp_repo_path: str = path.join(data_directory, temp_repo_directory)
    metrics_output_path: str = path.join(data_directory, metrics_output_directory)

    threads_num: int = min(max_threads, cpu_count())

    if not path.exists(temp_repo_path):
        print(f"Creating temporary repository directory: {temp_repo_path}")
        makedirs(temp_repo_path)

    if path.exists(metrics_output_path):
        print(f"Clearing the metrics output directory: {metrics_output_path}")
        for csv_file in glob.glob(path.join(metrics_output_path, "*.csv")):
            os.remove(csv_file)  # Supprimer chaque fichier CSV
    else:
        print(f"Creating metrics output directory: {metrics_output_path}")
        makedirs(metrics_output_path)
        
    if limit:
        versions = dict(list(versions.items())[:limit])

    multi_threaded_analysis(versions, threads_num)

    # Récupération des fichiers CSV et nettoyage des dossiers
    for version in versions.keys():
        version_temp_path = path.join(temp_repo_path, version)

        csv_files = glob.glob(path.join(version_temp_path, "*.csv"))
        if csv_files:
            metrics_csv = csv_files[0]
            shutil.copy(metrics_csv, path.join(metrics_output_path, f"{version}_metrics.csv"))
            print(f"Metrics for version {version} saved to {metrics_output_path}")
        else:
            print(f"No CSV file found for version {version} in {version_temp_path}")

        # Supprimer le répertoire de la version
        if path.exists(version_temp_path):
            rmtree(version_temp_path)
            print(f"Temporary repository for version {version} removed.")

    # Suppression du répertoire temporaire
    if path.exists(temp_repo_path):
        print(f"Removing the entire temporary repository: {temp_repo_path}")
        shutil.rmtree(temp_repo_path)