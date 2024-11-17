import os
import requests

from urllib.parse import quote


def hiveDL(command_file, jira_csv_dir, combined_csv_path, base_url, search_complement, query, temp_max=1000, start=0):
    """
    Download data from a Hive query in paginated form and save it to a CSV file.
    :param search_complement:
    :param base_url:
    :param command_file:
    :param jira_csv_dir:
    :param combined_csv_path:
    :param query:
    :param temp_max:
    :param start:
    :return:
    """
    # Delete the combined file if it exists
    if os.path.exists(combined_csv_path):
        os.remove(combined_csv_path)

    while True:
        # Build the paginated URL
        if start == 0:
            paginated_url = f"{base_url}{search_complement}{quote(query)}&tempMax={temp_max}"
        else:
            paginated_url = f"{base_url}{search_complement}{quote(query)}&pager/start={start}&tempMax={temp_max}"

        print(f"Fetching: {paginated_url}")
        response = requests.get(paginated_url)

        # Check answer status
        response.raise_for_status()

        # Check if we reached the end
        if not response.content.strip():  # Empty content means no more data
            print("No more data to fetch.")
            break

        # Save the data to a temporary file
        temp_file_path = os.path.join(jira_csv_dir, f"jira_data_{start}.csv")
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        # Combine the temporary file into the final CSV
        with open(temp_file_path, "r", encoding="utf-8") as temp_file:  # Force UTF-8 decoding
            header = start == 0  # Add the header only for the first file
            with open(combined_csv_path, "a", encoding="utf-8") as combined_file:  # Force UTF-8 encoding
                for i, line in enumerate(temp_file):
                    if i > 0 or header:  # Skip the header of the other files
                        combined_file.write(line)

        # Delete the temporary file
        os.remove(temp_file_path)

        # Update the start index
        start += temp_max

    # Save the query to a file
    with open(command_file, "w") as f:
        f.write(query)

    print(f"All data downloaded and saved to {combined_csv_path}")
