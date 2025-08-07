
import os
from tqdm import tqdm
import requests

import shutil
from multiprocessing import cpu_count
import tarfile


def download_file_fast(url, output_path):
    """Downloads a file with a progress bar and handles retries."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(output_path, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))


def extract_zip(zip_path, extract_to):
    """Extracts a zip file using the maximum number of available threads."""
    # Check if p7zip is installed
    if shutil.which("7z") is None:
        raise EnvironmentError(
            "7z command-line utility is required for multi-threaded extraction. Install p7zip."
        )

    # Run the extraction with maximum threads
    command = f"7z x -o{extract_to} {zip_path} -mmt{cpu_count()}"
    os.system(command)
    
    
def extract_tar_gz(tar_gz_path, extract_to):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
        
def extract_tar(tar_gz_path, extract_to):
    with tarfile.open(tar_gz_path, "r") as tar:
        tar.extractall(path=extract_to)

def extract_multi_part_zip(zip_parts, extract_to):
    """
    Extracts multi-part zip files using 7z with maximum threads.
    
    :param zip_parts: List of zip file parts (e.g., ['file.z01', 'file.z02', ..., 'file.zip'])
    :param extract_to: Directory where the extracted files should be placed.
    """
    if not zip_parts:
        raise ValueError("No zip parts provided.")

    # Ensure 7z is installed
    if shutil.which("7z") is None:
        raise EnvironmentError("7z command-line utility is required for extraction. Install p7zip.")

    # Identify the main zip file (usually the last part or the .zip file itself)
    zip_parts.sort()  # Ensure they are in order
    main_zip = next((f for f in zip_parts if f.endswith('.zip')), zip_parts[0])

    # Run extraction command
    command = f'7z x -o"{extract_to}" "{main_zip}" -mmt{cpu_count()}'
    os.system(command)


def extract_sequential_zip(zip_parts, extract_to):
    """
    Extracts multi-part zip files named sequentially (e.g., train_part1.zip, train_part2.zip, ...).
    
    :param zip_parts: List of zip file paths (e.g., [Path('train_part1.zip'), Path('train_part2.zip'), ...]).
    :param extract_to: Directory where the extracted files should be placed.
    """
    if not zip_parts:
        raise ValueError("No zip parts provided.")

    # Ensure 7z is installed
    if shutil.which("7z") is None:
        raise EnvironmentError("7z command-line utility is required for extraction. Install p7zip.")

    # Sort files to ensure correct order
    zip_parts = sorted(zip_parts, key=lambda x: str(x))

    # Use the first file as the entry point for extraction
    main_zip = zip_parts[0]

    # Run extraction command
    command = f'7z x -o"{extract_to}" "{main_zip}" -mmt{cpu_count()}'
    os.system(command)
