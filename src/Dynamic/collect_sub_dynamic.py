from os.path import basename
from collections import defaultdict


ALLOWED_EXTENSIONS = {".java", ".cpp"}

def is_allowed_file(file_path):
    """
    Check if the file has an allowed extension.

    Args:
        file_path (str): Full path of the file.
    
    Returns:
        bool: True if the file is allowed, False otherwise.
    """
    return any(file_path.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def collect_count_lines(commits):
    """
    Collect the total number of lines added and deleted in a range of commits.

    Args:
        commits (list): List of commits for the current version.

    Returns:
        dict: Lines added and deleted for each file.
    """
    lines_metrics = defaultdict(lambda: {"LinesAdded": 0, "LinesDeleted": 0})

    for commit in commits:
        for file_path, stats in commit.stats.files.items():
            if is_allowed_file(file_path):  # Filtre par extension
                file_name = basename(file_path)  # Ne garde que le nom du fichier
                lines_metrics[file_name]["LinesAdded"] += stats["insertions"]
                lines_metrics[file_name]["LinesDeleted"] += stats["deletions"]

    return lines_metrics


def collect_commit_count(commits):
    """
    Count the number of commits affecting each file.

    Args:
        commits (list): List of commits for the current version.

    Returns:
        dict: Commit count for each file.
    """
    commit_counts = defaultdict(int)

    for commit in commits:
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):  # Filtre par extension
                file_name = basename(file_path)  # Ne garde que le nom du fichier
                commit_counts[file_name] += 1

    return commit_counts


def collect_developer_count(commits):
    """
    Count the number of unique developers contributing to each file.

    Args:
        commits (list): List of commits for the current version.

    Returns:
        dict: Developer count for each file.
    """
    developer_counts = defaultdict(set)

    for commit in commits:
        author = commit.author.name  # Get the name of the author
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):  # Filter by extension
                file_name = basename(file_path)  # Keep only the file name
                developer_counts[file_name].add(author)

    # Convert sets to counts
    return {file: len(authors) for file, authors in developer_counts.items()}
