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


def collect_bug_fix_count(commits, keywords=["fix", "bug", "issue"]):
    """
    Count the number of bug-fixing commits affecting each file.

    Args:
        commits (list): List of commits for the current version.
        keywords (list): List of keywords to identify bug-fixing commits.

    Returns:
        dict: Bug-fixing commit count for each file.
    """
    bug_fix_counts = defaultdict(int)

    for commit in commits:
        if any(keyword in commit.message.lower() for keyword in keywords):
            for file_path in commit.stats.files.keys():
                if is_allowed_file(file_path):
                    file_name = basename(file_path)
                    bug_fix_counts[file_name] += 1

    return bug_fix_counts


def collect_cumulative_developer_count(commits):
    """
    Count the cumulative number of unique developers contributing to each file.

    Args:
        commits (list): List of all commits up to the current version.

    Returns:
        dict: Cumulative developer count for each file.
    """
    cumulative_developer_counts = defaultdict(set)

    for commit in commits:
        author = commit.author.name
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                cumulative_developer_counts[file_name].add(author)

    # Convert sets to counts
    return {file: len(authors) for file, authors in cumulative_developer_counts.items()}


def collect_average_time_between_changes(commits):
    """
    Calculate the average time between changes for each file.

    Args:
        commits (list): List of commits for the current version.

    Returns:
        dict: Average time in seconds between changes for each file.
    """
    time_diffs = defaultdict(list)

    for i in range(len(commits) - 1):
        current_commit = commits[i]
        next_commit = commits[i + 1]
        time_diff = (current_commit.committed_date - next_commit.committed_date)

        for file_path in current_commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                time_diffs[file_name].append(time_diff)

    return {
        file: sum(times) / len(times) if times else 0
        for file, times in time_diffs.items()
    }



def collect_comment_change_metrics(commits):
    """
    Count the number of commits that changed or did not change comments for each file.

    Args:
        commits (list): List of commits for the current version.

    Returns:
        dict: Counts of comment-changing and non-comment-changing commits for each file.
    """
    comment_metrics = defaultdict(lambda: {"ChangedComments": 0, "UnchangedComments": 0})

    for commit in commits:
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                # Simplification : on suppose que les commentaires sont détectés via un mot-clé
                if "comment" in commit.message.lower():  # Exemple simple
                    comment_metrics[file_name]["ChangedComments"] += 1
                else:
                    comment_metrics[file_name]["UnchangedComments"] += 1

    return comment_metrics


def collect_cumulative_commit_count(commits):
    """
    Count the cumulative number of commits affecting each file.

    Args:
        commits (list): List of all commits up to the current version.

    Returns:
        dict: Cumulative commit count for each file.
    """
    cumulative_commit_counts = defaultdict(int)

    for commit in commits:
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                cumulative_commit_counts[file_name] += 1

    return cumulative_commit_counts

def collect_cumulative_comment_metrics(commits):
    """
    Count cumulative comment-changing and non-comment-changing commits for each file.

    Args:
        commits (list): List of all commits up to the current version.

    Returns:
        dict: Cumulative counts of comment-changing and non-comment-changing commits for each file.
    """
    cumulative_comment_metrics = defaultdict(lambda: {"ChangedComments": 0, "UnchangedComments": 0})

    for commit in commits:
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                # Simplification : on suppose que les commentaires sont détectés via un mot-clé
                if "comment" in commit.message.lower():  # Exemple simple
                    cumulative_comment_metrics[file_name]["ChangedComments"] += 1
                else:
                    cumulative_comment_metrics[file_name]["UnchangedComments"] += 1

    return cumulative_comment_metrics

def collect_cumulative_average_time_between_changes(commits):
    """
    Calculate the cumulative average time between changes for each file.

    Args:
        commits (list): List of all commits up to the current version.

    Returns:
        dict: Cumulative average time in seconds between changes for each file.
    """
    time_diffs = defaultdict(list)

    for i in range(len(commits) - 1):
        current_commit = commits[i]
        next_commit = commits[i + 1]
        time_diff = (current_commit.committed_date - next_commit.committed_date)

        for file_path in current_commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                time_diffs[file_name].append(time_diff)

    return {
        file: sum(times) / len(times) if times else 0
        for file, times in time_diffs.items()
    }


def collect_average_expertise(commits):
    """
    Calculate the average expertise of developers for each file.

    Args:
        commits (list): List of all commits up to the current version.

    Returns:
        dict: Average expertise for each file.
    """
    developer_commit_counts = defaultdict(lambda: defaultdict(int))

    # Collect the number of commits per developer for each file
    for commit in commits:
        author = commit.author.name
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                developer_commit_counts[file_name][author] += 1

    # Calculate average expertise for each file
    average_expertise = {}
    for file, dev_counts in developer_commit_counts.items():
        counts = list(dev_counts.values())
        average_expertise[file] = sum(counts) / len(counts) if counts else 0

    return average_expertise


def collect_minimum_expertise(commits):
    """
    Calculate the minimum expertise of developers for each file.

    Args:
        commits (list): List of all commits up to the current version.

    Returns:
        dict: Minimum expertise for each file.
    """
    developer_commit_counts = defaultdict(lambda: defaultdict(int))

    # Collect the number of commits per developer for each file
    for commit in commits:
        author = commit.author.name
        for file_path in commit.stats.files.keys():
            if is_allowed_file(file_path):
                file_name = basename(file_path)
                developer_commit_counts[file_name][author] += 1

    # Calculate minimum expertise for each file
    minimum_expertise = {}
    for file, dev_counts in developer_commit_counts.items():
        counts = list(dev_counts.values())
        minimum_expertise[file] = min(counts) if counts else 0

    return minimum_expertise
