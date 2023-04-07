from typing import List
import os


def find_directories(root_dir: str, file_extension: str) -> List[str]:
    """
    Recursively searches for directories containing files with the given file extension in the given root directory and
     its subdirectories.
    Args:
        root_dir (str): The root directory to start searching from.
        file_extension (str): The file extension to search for (e.g., ".txt").
    Returns:
         A list of absolute paths to the directories containing files with the given file extension.
    """
    found_directories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_extension):
                found_directories.append(os.path.abspath(root))
                break
    return found_directories
