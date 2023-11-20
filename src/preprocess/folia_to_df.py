import folia.main as folia
import pandas as pd
import os
import random 
import re
from preprocess.dbnl_xml_to_txt import extract_dbnl_id

def split_words(sentence):
    """
    Splits a sentence in words.
    """
    return [w.text() for w in sentence.words()]

def double_split_folia(doc):
    """
    Splits a folia file in sentences and the sentences in words.
    """
    return [split_words(s) for s in doc.sentences()]

def single_split_folia(doc):
    """
    Splits a folia file in sentences.
    """
    return [s for s in doc.sentences()]

def select_random_files(path_to_folder, file_extension, num_files):
    """
    Randomly selects the specified number of files of a specified extension.
    If number of files exceeds the available number of files, num_files 
    is scaled back and the user is notified.
    """
    try:
        file_paths = [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith(file_extension)]
        if num_files > len(file_paths):
            print(f"Warning: Requested {num_files} files, but only {len(file_paths)} files are available.")
            num_files = len(file_paths)
        return random.sample(file_paths, num_files)
    except OSError as e:
        print(f"Error accessing directory: {e}")
        return []


def select_all_files(path_to_folder, file_extension):
  """
  Selects all files with a specified extension in a specified folder.
  """
  try: 
     return [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith(file_extension)]
  except OSError as e:
        print(f"Error accessing directory: {e}")
        return []

def folia_to_df(folia_path):
  """
  Converts a folia file to a df, with columns for the dbnl_id,
  the intact sentences and the sentences split up in words.
  """
  try:
    doc = folia.Document(file=folia_path)
    sentence_divided_lst = single_split_folia(doc)
    word_divided_lst = double_split_folia(doc)
    df = pd.DataFrame({'sentences': sentence_divided_lst,'split': word_divided_lst})
    df["text_id"] = extract_dbnl_id(folia_path)
    return df
  except Exception as e:
    print(f"Error processing file {folia_path}: {e}")
    return pd.DataFrame()
