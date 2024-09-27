from typing import Dict, List, Any
import yaml
import os
from PIL import Image


def create_folder_structure(root_folder: str, output_file: str) -> None:
    """
    Scan a folder for subfolders and save the structure to a YAML file.

    Args:
        root_folder (str): Path to the root folder to scan.
        output_file (str): Path where the output YAML file will be saved.

    Raises:
        FileNotFoundError: If the root_folder does not exist.
        PermissionError: If there's no write permission for the output_file.
    """
    folder_structure: Dict[str, List[str]] = {}

    if not os.path.isdir(root_folder):
        raise FileNotFoundError(
            f"Root folder '{root_folder}' doesn't exist."
        )

    for subfolder in os.scandir(root_folder):
        if subfolder.is_dir():
            png_files: List[str] = [
                os.path.join(subfolder.path, file.name)
                for file in os.scandir(subfolder.path)
                if file.is_file() and file.name.lower().endswith('.png')
            ]
            folder_structure[subfolder.name] = png_files

    try:
        with open(output_file, 'w') as f:
            yaml.dump(folder_structure, f, default_flow_style=False)
    except PermissionError:
        raise PermissionError(
            f"No write permission for the output file '{output_file}'."
        )


def read_img(img_path: str) -> Image.Image:
    """
    Read an image file and return a PIL Image object.

    Args:
        img_path (str): The path to the image file.

    Returns:
        Image.Image: A PIL Image object representing the loaded image.

    Raises:
        FileNotFoundError: If the specified image file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.
    """
    return Image.open(img_path)


def to_yaml(dictionary: Dict[Any, Any], path: str) -> None:
    """
    Write a dictionary to a YAML file.

    Args:
        dictionary (Dict[Any, Any]): The dictionary to be written to the YAML file.
        path (str): The path where the YAML file will be created or overwritten.

    Raises:
        IOError: If there's an error writing to the file.
        yaml.YAMLError: If there's an error in dumping the dictionary to YAML format.
    """
    with open(path, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)


def from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        yaml_path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(yaml_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The yaml file was not found in the current directory."
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing the YAML file: {e}"
        )


def clean_string(string: str) -> str:
    """
    Clean and format a string by removing newlines, extra spaces, and leading colons.

    Args:
        string (str): The input string to be cleaned.

    Returns:
        str: The cleaned and formatted string.
    """
    string = string.replace('\n', '').strip()
    string = ' '.join(string.split())
    return string[2:] if string.startswith(': ') else string
