""" 
"""
import folia.main as folia
import pandas as pd
import os
import random 
from preprocess.dbnl_xml_to_txt import extract_dbnl_id

def list_folia_files(directory):
  """ 
  List all folia files in a certain directory.
  """
  folia_files = []
  for filename in os.listdir(directory):
    if filename.endswith('folia.xml'):
      folia_files.append(filename)
  return folia_files

def split_words(sentence):
  sentence_lst = []
  for w in sentence.words():
    sentence_lst.append(w.text())
  return sentence_lst

def split_folia(doc):
  doc_lst = []
  for s in doc.sentences():
    s_lst = split_words(s)
    doc_lst.append(s_lst)
  return doc_lst

def select_random_files(path_to_folder, file_extension, num_files):
  file_paths = [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith(file_extension)]
  return random.sample(file_paths, num_files)
  
def select_all_files(path_to_folder, file_extension):
  return [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith(file_extension)]


def folia_to_df(folia_path):
  doc = folia.Document(file=folia_path)
  doc_lst = split_folia(doc)
  df = pd.DataFrame({'sentence': doc_lst})
  df["text_id"] = extract_dbnl_id(folia_path)
  return df

