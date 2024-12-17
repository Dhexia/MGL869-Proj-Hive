from packaging.version import parse as version_parse

def build_dependencies(filtered_versions):
    """
    Builds enriched dependencies between tagged versions, including dates and branch origins.

    Args:
        filtered_versions (dict): Dictionary of tagged versions with their Git commit.
                                  Format: {'version': <git.Commit>}
    
    Returns:
        dict: Dictionary of enriched dependencies sorted by ascending version with:
              - "previous": Parent version
              - "next": Closest child version
              - "date": Commit date
              - "commit": The Git commit associated with the version
              - "branch_origin": Version where a new branch originates
    """
    dependencies = {
        version: {
            "previous": None,
            "next": None,  # Single closest child
            "date": commit.committed_datetime.strftime("%Y-%m-%d"),
            "commit": commit,  # Store the commit object
            "branch_origin": None,
        }
        for version, commit in filtered_versions.items()
    }

    sorted_versions = sorted(filtered_versions.items(), key=lambda x: x[1].committed_date)

    for i, (current_version, current_commit) in enumerate(sorted_versions):
        for next_version, next_commit in sorted_versions[i + 1:]:
            if next_commit in current_commit.parents:
                # Link to the closest child (single "next")
                dependencies[current_version]["next"] = next_version
                dependencies[next_version]["previous"] = current_version
                break

    # Mark new branches and add additional information
    for version, data in dependencies.items():
        if data["previous"] is None and version != sorted_versions[0][0]:

            # Identify the branch origin
            branch_origin = next(
                (v for v, d in dependencies.items() if version == d["next"]),
                None,
            )
            data["branch_origin"] = branch_origin

    # Sort by ascending version
    sorted_dependencies = dict(
        sorted(dependencies.items(), key=lambda x: version_parse(x[0]))
    )

    return adjust_branches(sorted_dependencies)


def adjust_branches(dependencies):
    """
    Adjusts floating branches by inserting `previous: None` versions 
    into the correct place in the dependency hierarchy.
    
    Args:
        dependencies (dict): Dictionary of existing dependencies.
    
    Returns:
        dict: Corrected dependencies with updated hierarchy.
    """
    # Find floating branches (previous: None but not the main root)
    floating_branches = [
        version
        for version, data in dependencies.items()
        if data["previous"] is None and version != "main"
    ]

    for branch in floating_branches:
        branch_date = dependencies[branch]["date"]
        branch_version = version_parse(branch)
        best_parent = None

        # Find the best parent for this branch
        for candidate, data in dependencies.items():
            if candidate == branch:
                continue
            candidate_version = version_parse(candidate)
            candidate_date = data["date"]

            # Parent conditions:
            # - The parent's date is earlier than the branch's date
            # - The parent's version is earlier than the branch's version
            if candidate_date < branch_date and candidate_version < branch_version:
                # Check if it's the best parent (closest version)
                if best_parent is None or version_parse(best_parent) < candidate_version:
                    best_parent = candidate

        # Add this branch as a child of the best parent
        if best_parent:
            if not dependencies[best_parent]["next"]:
                dependencies[best_parent]["next"] = []
            dependencies[best_parent]["next"].append(branch)
            dependencies[branch]["previous"] = best_parent

    # Sort the `next` field by version
    for version, data in dependencies.items():
        if isinstance(data["next"], list):
            data["next"].sort(key=lambda x: version_parse(x))

    return dependencies


def display_hierarchy(dependencies):
    """
    Display the hierarchical tree of dependencies in a readable format.
    
    Args:
        dependencies (dict): The dependency tree.
    
    Returns:
        None: Prints the hierarchical tree.
    """
    visited = set()  # To avoid processing a version multiple times

    def traverse(version, level=0):
        # Avoid cycles or revisiting nodes
        if version in visited:
            return
        visited.add(version)

        data = dependencies[version]
        commit = data.get("commit")
        commit_hash = commit.hexsha[:7] if commit else "N/A"  # Handle Commit objects safely
        print("    " * level + f"{version} ({data['date']}) [Commit: {commit_hash}]")

        # Ensure `next` is a list before iterating
        for child in data.get("next") or []:
            traverse(child, level + 1)

    # Start traversal from the root nodes (those with `previous=None`)
    roots = [v for v, d in dependencies.items() if d.get("previous") is None]
    for root in sorted(roots, key=lambda x: version_parse(x)):
        traverse(root)

