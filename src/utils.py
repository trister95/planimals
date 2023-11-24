import ast
from langdetect import detect
import os
import glob
import urllib.request
import zipfile
import requests
import re


def create_if_absent(directory):
    """
    Function creates a directory if it doesn't exist already.
    """
    import os

    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory", directory, "created")
    else:
        print("Directory", directory, "already exists")
    return


def get_lst_from_csv(value):
    """
    Would be smart to do some more checking,
    since I expected that the first value declaratiod
    wouldn't be needed (but it turns out it is).
    """
    value = ast.literal_eval(value)
    value_str = value.decode("utf-8")
    return ast.literal_eval(value_str)


def has_letter(s):
    # Helper function to check if a string has a letter
    return any(char.isalpha() for char in s)


def find_language_in_ucto_tokenized_sentence(sentence_lst):
    return detect((" ".join(sentence_lst)))


def list_extension_files(directory, extension):
    """
    Outputs all files with a specific extension in a directory.
    """
    try:
        # Using glob for pattern matching
        return [
            os.path.basename(file) for file in glob.glob(f"{directory}/*.{extension}")
        ]
    except OSError as e:
        print(f"Error: {e}")
        return []


def download_and_unzip(url, output_dir="."):
    # Download the zipfile
    filename, _ = urllib.request.urlretrieve(url)

    # Unzip the zipfile
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    return


def download(url, output):
    response = requests.get(url)
    with open(output, "wb") as file:
        file.write(response.content)
    return


def extract_dbnl_id(f_name):
    """
    Etract dbnl-id from a filename.
    """
    pattern = r"[a-zA-Z_]{4}\d{3}[a-zA-Z_\d]{4}\d{2}"
    regex = re.compile(pattern)
    match = regex.search(f_name)
    return match[0] if match else None


def clean_whitespace(text):
    """Clean unnecessary whitespaces from the text."""
    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text)
    # Optional: Convert multiple newlines to a double newline for paragraph separation
    text = re.sub(r"(\s*\n\s*)+", "\n\n", text)
    return text.strip()

def count_files_in_folder(folder_path):
    """
    Couns file in a directory. 
    Input: path to directory (string)
    """
    file_count = 0

    # Iterate over all the items in the folder
    for _, _, files in os.walk(folder_path):
        file_count += len(files)

    return file_count