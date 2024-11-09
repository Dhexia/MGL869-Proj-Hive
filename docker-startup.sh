#!/bin/bash

# Path to the main directory to check
DIR_PATH="/Hive/understand_app"
SCITOOLS_DIR="$DIR_PATH/scitools"



# Retrieve environment variables
# Exit with an error if either of the variables is not set
if [ -z "$UNDERSTAND_VERSION" ] || [ -z "$UNDERSTAND_BUILD" ]; then
    echo "Error: UNDERSTAND_VERSION and UNDERSTAND_BUILD must be set in the environment."
    exit 1
fi

# URL and file name for the Understand archive
ARCHIVE_URL="http://builds.scitools.com/all_builds/b${UNDERSTAND_BUILD}/Understand/Understand-CLI-${UNDERSTAND_VERSION}.${UNDERSTAND_BUILD}-Linux-64bit.tgz"
ARCHIVE_NAME="Understand-CLI-${UNDERSTAND_VERSION}.${UNDERSTAND_BUILD}-Linux-64bit.tgz"
ARCHIVE_PATH="$DIR_PATH/$ARCHIVE_NAME"

# Check if the "scitools" directory exists, if not, extract the archive
if [ -d "$SCITOOLS_DIR" ]; then
    echo "The directory $SCITOOLS_DIR already exists. No extraction needed."
else
    # If the scitools directory doesn't exist, check if the archive is present
  if [ ! -f "$ARCHIVE_PATH" ]; then
    # If the archive doesn't exist, download it
    echo "Archive $ARCHIVE_NAME not found. Downloading from $ARCHIVE_URL..."

    # Download the archive
    wget -q -P "$DIR_PATH" "$ARCHIVE_URL"

    # Check if the download was successful
    if [ ! -f "$ARCHIVE_PATH" ]; then
        echo "Failed to download the archive. Please check the URL or network connection."
        exit 1
    fi

    echo "Archive downloaded successfully."
  else
      echo "Archive $ARCHIVE_NAME found."
  fi

  echo "Extracting..."

  # Extract the archive
  tar -xzf "$ARCHIVE_PATH" -C "$DIR_PATH"
  # Remove the archive after extraction
  rm "$ARCHIVE_PATH"
  echo "Extraction completed and $ARCHIVE_NAME has been removed."

    # License activation should happen only on the first extraction
    echo "First time extraction detected. Activating the license..."

fi

# Function to set the license
set_license() {
    echo "Setting the license code..."
    "$SCITOOLS_DIR/bin/linux64/und" -setlicensecode "$LICENSE_CODE"
    if [ $? -eq 0 ]; then
        echo "License successfully set in Understand."
        echo "$LICENSE_HASH" > "$LICENSE_FILE"
        echo "License saved to $LICENSE_FILE"
    else
        echo "Failed to set the license in Understand. Please check your license code and the number of activations."
        exit 1
    fi
}

if [ -z "$UNDERSTAND_LICENSE" ]; then
    echo "Error: UNDERSTAND_LICENSE must be set in the environment."
    exit 1
fi

LICENSE_HASH=$(echo -n "$UNDERSTAND_LICENSE" | sha256sum | awk '{print $1}')
LICENSE_FILE="$DIR_PATH/license"

# Check if the license file exists
if [ -f "$LICENSE_FILE" ]; then
    STORED_LICENSE=$(cat "$LICENSE_FILE")
    if [ "$STORED_LICENSE" != "$LICENSE_HASH" ]; then
        echo "License mismatch detected. Attempting to update the license..."
        set_license
    else
        echo "The current license matches the stored license. No changes needed."
    fi
else
    echo "License file not found. Adding the license..."
    set_license
fi

# Start jupyter notebook

jupyter notebook --allow-root --no-browser --NotebookApp.token='my-token'
