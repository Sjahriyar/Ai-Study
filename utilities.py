import os
import urllib.request
import tarfile
from pathlib import Path

# Define the URL and local file paths
file_name = "mnist_sample.tgz"
extract_path = "./mnist_sample"  # Directory where the contents will be extracted

# Function to download and extract the data
def untar_data(url, dest_file, extract_dir):
    # Step 1: Download the file if it doesn't exist
    if not os.path.exists(dest_file):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_file)
        print(f"Download completed: {dest_file}")
    else:
        print(f"{dest_file} already exists. Skipping download.")

    # Step 2: Extract the file if the directory doesn't exist
    if not os.path.exists(extract_dir):
        print(f"Extracting {dest_file}...")
        with tarfile.open(dest_file, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"Extraction completed: {extract_dir}")
    else:
        print(f"{extract_dir} already exists. Skipping extraction.")
    
    # Return the path to the extracted directory
    return Path(extract_dir)