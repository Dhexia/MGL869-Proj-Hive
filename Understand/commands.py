from os import path
from subprocess import Popen, PIPE
from configparser import ConfigParser

# Read the config file -------------------------------------------------- #

config = ConfigParser()
config.read('config.ini')

# Global variables ------------------------------------------------------ #

hive_git_directory: str = config['GIT']["HiveGitDirectory"]
hive_repo_name: str = config['GIT']["HiveGitRepoName"]
understand_project_name: str = config["UNDERSTAND"]["UnderstandProjectName"]
und: str = config["UNDERSTAND"]["UnderstandCommand"]
understand_metrics_file_name: str = config["UNDERSTAND"]["UnderstandMetricsFileName"]
data_directory: str = config["GENERAL"]["DataDirectory"]

und_project_path: str = path.join(data_directory, hive_git_directory, understand_project_name)
und_metrics_path: str = path.join(hive_git_directory, understand_project_name[:-4:] + ".csv")
hive_git_repo_dir: str = path.join(hive_git_directory, hive_repo_name)


# Function to run a command --------------------------------------------- #

def run_command(command: str):
    command_args: [str] = command.split(" ")
    print(f"Running command : \n     {command}")
    process = Popen(command_args, stdout=PIPE, stderr=PIPE).communicate()[0]
    print(process.decode("utf-8"))


# Understand commands as Strings ---------------------------------------- #

def und_create_command(project_path=und_project_path) -> None:
    return run_command(f"{und} create -db {project_path} -languages Java c++")


def und_purge_command() -> None:
    return run_command(f"{und} purge -db {und_project_path}")


def und_add_command(repo_path=hive_git_repo_dir, project_path=und_project_path) -> None:
    return run_command(f"{und} add {repo_path} -db {project_path}")


def und_analyze_command(project_path=und_project_path) -> None:
    return run_command(f"{und} analyze -db {project_path} -quiet")


def und_analyze_changes_command(project_path=und_project_path) -> None:
    return run_command(f"{und} analyze -db {project_path} -quiet -rescan -changed")


def und_metrics_command(project_path=und_project_path) -> None:
    return run_command(f"{und} metrics {project_path}")
