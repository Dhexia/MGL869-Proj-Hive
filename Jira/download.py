from configparser import ConfigParser
from os import path, listdir, makedirs, remove
from urllib.parse import quote
from requests import get as http_get
from pandas import read_csv, DataFrame


def jira_download() -> DataFrame or None:
    """
    This function downloads the data from JIRA if needed
    function uses config file to get the credentials and other details
    :return:
    """

    # Read the config file -------------------------------------------------- #

    config = ConfigParser()
    config.read('config.ini')
    section = 'JIRA'

    # Variables ------------------------------------------------------------- #

    base_url: str = config[section]["BaseUrl"]
    search_complement: str = config[section]["SearchComplement"]
    query: str = config[section]["Query"]
    jira_csv_directory: str = config[section]["JiraCSVDirectory"]
    query_each_run: str = config[section]["QueryEachRun"]
    jira_combined_csv: str = config[section]["JiraCombinedCSV"]

    data_directory: str = config["GENERAL"]["DataDirectory"]

    jira_csv_path: str = path.join(data_directory, jira_csv_directory)
    command_file: str = path.join(jira_csv_path, "command.txt")
    combined_csv_path: str = path.join(jira_csv_path, jira_combined_csv)

    max_data_lines: int = 1000
    start: int = 0

    url: str = f"{base_url}{search_complement}{quote(query)}"

    # Check if the data is already downloaded ------------------------------ #

    if not path.exists(jira_csv_path):
        makedirs(jira_csv_path)
    else:
        csv_files = [f for f in listdir(jira_csv_path) if f.endswith(".csv")]
        if jira_combined_csv in csv_files and query_each_run != "Yes":
            print("Data already downloaded")
            print(f"Filter = '{query}'")
            return read_csv(combined_csv_path, low_memory=False)

    # Delete the combined file if it exists -------------------------------- #

    if path.exists(combined_csv_path):
        remove(combined_csv_path)

    # Download the data ---------------------------------------------------- #

    print(f"Downloading data from {base_url}...")
    print(f"Filter = '{query}'")
    while True:
        # Build the paginated URL
        if start == 0:
            paginated_url = f"{url}&tempMax={max_data_lines}"
        else:
            paginated_url = f"{url}&pager/start={start}&tempMax={max_data_lines}"

        print(f"Fetching: {start} -> {start + max_data_lines - 1}")
        response = http_get(paginated_url)
        response.raise_for_status()

        # Check if we reached the end
        if not response.content.strip():  # Empty content means no more data
            print("No more data to fetch.")
            break

        # Save the data to a temporary file
        temp_file_path = path.join(jira_csv_path, f"jira_data_{start}.csv")
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        # Combine the temporary file into the final CSV
        with open(temp_file_path, "r", encoding="utf-8") as temp_file:
            # Add the header only for the first file
            header = start == 0

            with open(combined_csv_path, "a", encoding="utf-8") as combined_file:
                for i, line in enumerate(temp_file):
                    if i > 0 or header:
                        # Skip the header of the other files
                        combined_file.write(line)

        # Delete the temporary file
        remove(temp_file_path)

        # Update the start index
        start += max_data_lines

    # Save the query to a file
    with open(command_file, "w") as f:
        f.write(query)

    print(f"All data downloaded and saved to {combined_csv_path}")
    return read_csv(combined_csv_path, low_memory=False)
