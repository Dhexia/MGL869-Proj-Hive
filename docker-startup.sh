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
    "$SCITOOLS_DIR/bin/linux64/und" -setlicensecode "$UNDERSTAND_LICENSE" >> /Hive/app/license.log
    if [ $? -eq 0 ]; then
        echo "License successfully set in Understand." >> /Hive/app/license.log
    else
        echo "Failed to set the license in Understand. Please check your license code and the number of activations." >> /Hive/app/license.log
        exit 1
    fi
}

if [ -z "$UNDERSTAND_LICENSE" ]; then
    echo "Error: UNDERSTAND_LICENSE must be set in the environment."
    echo "Error: UNDERSTAND_LICENSE must be set in the environment." > /Hive/app/license.log
    exit 1
fi

echo "" > /Hive/app/license.log

set_license

cleanup() {
    echo "Container stopped, performing cleanup..." >> /Hive/app/license.log
    # Unregister the license
    "$SCITOOLS_DIR/bin/linux64/und" -deregisterlicensecode >> /Hive/app/license.log
    exit 0
}

trap 'cleanup' SIGTERM
trap 'cleanup' SIGINT


# Start Jupyter Notebook in the background
echo "Starting Jupyter Notebook..."
jupyter notebook --allow-root --no-browser --NotebookApp.token='my-token' &

#Wait
wait $!
