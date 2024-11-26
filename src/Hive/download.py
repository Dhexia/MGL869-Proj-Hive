from configparser import ConfigParser
from os import path, makedirs
from git import Repo, GitCommandError


def git_download() -> Repo:
    """
    This function clones the git repository if needed
    :return: Repo
    """
    # Read the config file -------------------------------------------------- #

    config = ConfigParser()
    config.read('config.ini')

    # Variables ------------------------------------------------------------- #

    hive_git_directory: str = config["GIT"]["HiveGitDirectory"]
    hive_git_repo_name: str = config["GIT"]["HiveGitRepoName"]
    hive_git_url: str = config["GIT"]["HiveGitUrl"]
    hive_git_always_clone: bool = config["GIT"]["HiveGitAlwaysClone"] != "No"
    hive_git_always_pull: bool = config["GIT"]["HiveGitAlwaysPull"] == "Yes"

    data_directory: str = config["GENERAL"]["DataDirectory"]

    full_path: str = path.join(data_directory, hive_git_directory, hive_git_repo_name)
    clone: bool = False

    # Check if the data is already downloaded ------------------------------ #

    print(full_path, not path.exists(full_path))
    if not path.exists(full_path):
        print(f"Creating the directory: {full_path}")
        makedirs(full_path)
        clone = True

    if not path.exists(path.join(full_path, ".git")):
        clone = True

    # Clone the repository ------------------------------------------------- #

    if clone or hive_git_always_clone:
        print(f"Cloning the repository: {hive_git_url}")
        repo = Repo.clone_from(hive_git_url, full_path)
    else:
        repo = Repo(full_path)
        if hive_git_always_pull:
            try:
                print(f"Pulling the repository: {hive_git_url}")
                repo.remotes.origin.pull()
            except GitCommandError as e:
                print("Pull failed : ", e)

    return repo
