import os
import gdown
import zipfile

def download_data(url, output_zip):
    if not os.path.exists(output_zip):
        print(f"Downloading data from {url}...")
        gdown.download(url, output_zip, quiet=False,fuzzy = True)
    else:
        print(f"{output_zip} already exists, skipping download.")

def extract_data(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed.")
    else:
        print(f"{extract_dir} already exists, skipping extraction.")

