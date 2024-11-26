from packaging import version

def filter_versions_by_min(versions, releases_regex, min_version):
    """
    Filters and sorts versions based on a minimum version requirement.
    
    :param versions: dict of version strings mapped to their associated data.
    :param releases_regex: list of compiled regex patterns to match version strings.
    :param min_version: str, the minimum version to include in the results.
    :return: dict of filtered and sorted versions, and the count of these versions.
    """
    # Validate min_version to ensure it's not below "1.0"
    if version.parse(min_version) < version.parse("1.0"):
        raise ValueError("The minimum version cannot be less than '1.0'.")

    filtered_versions = {}

    # Iterate through the versions and apply regex filtering and minimum version check
    for version_str in versions:  
        if any(regex.match(version_str) for regex in releases_regex):
            # Extract the version number part from the string
            version_numbers = version_str.split("-")[1]
            # Compare with the minimum version
            if version.parse(version_numbers) >= version.parse(min_version):
                filtered_versions[version_numbers] = versions[version_str]

    # Sort the filtered versions by their committed date in descending order
    filtered_versions = dict(sorted(
        filtered_versions.items(),
        key=lambda item: item[1].committed_datetime,
        reverse=True
    ))

    return filtered_versions

def sort_filtered_versions(filtered_versions):
    """
    Sort the filtered_versions dictionary by version in ascending order.

    Args:
        filtered_versions (dict): Dictionary with versions as keys and commits as values.

    Returns:
        dict: Sorted dictionary with versions in ascending order.
    """

    # Sort by version number in descending order
    sorted_versions = dict(sorted(filtered_versions.items(), key=lambda item: version.parse(item[0])))

    return sorted_versions