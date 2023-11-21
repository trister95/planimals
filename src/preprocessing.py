import os
import lxml.etree as ET
import html.entities
import ucto
import tqdm
import folia.main as folia
import pandas as pd
import random 
from transformers import AutoTokenizer
from utils import extract_dbnl_id, list_extension_files, clean_whitespace, find_language_in_ucto_tokenized_sentence, has_letter


"""
These functions extract the actual literary texts
from a directory with dbnl-xml files.
"""

def make_html_entity_dict():
  """
  This function makes a dict where html character entities
  (like: "&nbsp;") are coupled to xml-readable html decimal
  representations (like: "&160;").

  Note that in this function the named entities cannot
  end with a semicolon and can only have a length of 1.
  """
  html_decimal_dict = {}

  for entity, codepoint in html.entities.html5.items():
    if len(codepoint) == 1:
      if entity[-1] != ";":
        html_decimal_dict[entity] = ord(codepoint)
  
  return html_decimal_dict
    
def additional_declaration_str(html_decimal_dict, recognition_str ):
  """
  This function formulates a additional declaration string for XML documents.
  Give it the end of the old declaration (make sure this only pops up once in
  the document!) and provide a dictionary.

  Note that this function is written for conversion of html character entities
  to numerical representations. For other use, some tweaks might be needed.
  """
  new_declaration = ""

  for item in html_decimal_dict.items():
    new_str = f'<!ENTITY {item[0]} "&#{item[1]};">\n'
    new_declaration += new_str

  return  recognition_str+ '\n[' + new_declaration + ']'

def add_declaration_to_xml(xml_dir, start_doctype, end_doctype, addition):
  """
  This functions adds an entity declaration. It also checks if the "new" declaration
  isn't already present.
  """
  for f_name in os.listdir(xml_dir):
    if f_name.endswith('.xml'):
      with open(os.path.join(xml_dir, f_name), 'r', encoding = 'utf-8') as f:
        xml_text = f.read()
        start = xml_text.find(start_doctype)
        end = xml_text.find(end_doctype) + len(end_doctype)
        doctype = xml_text[start:end]
        new_doctype = doctype.replace(end_doctype, addition)
        if new_doctype not in xml_text:
          xml_text = xml_text[:start] + new_doctype + xml_text[end:]
          with open(os.path.join(xml_dir, f_name), 'w', encoding = 'utf-8') as f:
            f.write(xml_text)
  return

def list_xml_files(directory):
    """ 
    Lists all the xml files in a directory.
    """
    xml_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_files.append(filename)
    return xml_files

def extract_notes(root):
    """Extract and store notes from the XML tree."""
    notes = []
    for note in root.findall('.//note'):
        notes.append(clean_whitespace(''.join(note.itertext())))
        note.clear()  # Remove note from the main tree
    return notes

def find_text_element(root):
    """Find and return the <text> element from the XML tree."""
    text_elements = root.findall('.//text')
    if len(text_elements) == 0:
        raise ValueError("No <text> element found in the XML file.")
    elif len(text_elements) > 1:
        raise ValueError("More than 1 <text> element found in the XML file.")
    return text_elements[0]

def combine_text(main_text, notes, include_notes):
    """Combine the main text and notes based on the chosen option."""
    if include_notes == 'end':
        return main_text + '\n\nNotes:\n' + '\n'.join(notes)
    elif include_notes == 'only':
        return '\n'.join(notes)
    else:  # Default, only main text
        return main_text

def get_text_dbnl(xml_file, id, output_dir, include_notes='default'):
    """Extracts text from a dbnl xml-file and saves it as a .txt file."""
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract and store notes
    notes = extract_notes(root)

    # Find the <text> element and extract main text
    text_element = find_text_element(root)
    main_text = clean_whitespace(''.join(text_element.itertext()))

    # Combine text based on the chosen option
    combined_text = combine_text(main_text, notes, include_notes)

    # Save the combined text to a file
    with open(f'{output_dir}/{id}.txt', 'w', encoding='utf-8') as f:
        f.write(combined_text)

def dbnl_to_txt(input_dir, output_dir, include_notes="default"):
    """ Puts the pieces of the pipeline together. """
    xml_files = list_xml_files(input_dir)
    for f in xml_files:
        id = extract_dbnl_id(f)
        p = os.path.join(input_dir, f)
        try:
            get_text_dbnl(p, id, output_dir, include_notes=include_notes)
        except Exception as e:
            error_message = f"file {id} could not be parsed: {e}"
            print(error_message)
    return

"""
This function is for changing txt files to folia files.
"""

def txt_to_folia(input_dir, output_dir):
    """
    Tokenizes txt files by converting them to folia.xml files.
    """
    configurationfile_ucto = "tokconfig-nld-historical"
    tokenizer = ucto.Tokenizer(configurationfile_ucto, foliaoutput=True)

    for f in tqdm.tqdm(list_extension_files(input_dir, "txt"), desc="Looping through text files"):
        try:
            input_path = os.path.join(input_dir, f)
            output_filename = f"{extract_dbnl_id(f)}.folia.xml"
            out_path = os.path.join(output_dir, output_filename)
            
            if not os.path.exists(out_path):
                tokenizer.tokenize(input_path, out_path)
        except Exception as e:
            print(f"An error occurred with file {f}: {e}")

"""
These functions are for changing folia files to
df formats.

"""
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
    Converts a folia file to a DataFrame, with columns for the dbnl_id,
    the intact sentences and the sentences split up in words.
    Only includes sentences that contain letters and are in Dutch.
    """
    try:
        doc = folia.Document(file=folia_path)
        sentences = []
        for s in doc.sentences():
            sentence_text = s.text()
            if has_letter(sentence_text) and find_language_in_ucto_tokenized_sentence(sentence_text.split()) == 'nl':
                sentences.append(sentence_text)

        df = pd.DataFrame({'sentences': sentences})
        df["text_id"] = extract_dbnl_id(folia_path)
        return df
    except Exception as e:
        print(f"Error processing file {folia_path}: {e}")
        return pd.DataFrame()

def join_meta_csv(df, csv_path, left_on="text_id", right_on="ti_id", sep = ","):
    """
    Add metadata to a dataframe with dbnl sentences.
    The metadata is connected based on the dbnl-id
    of the text. 

    NB: I use drop_duplicates here, this might make the metadata
    incomplete on some aspects.
    """
    metadata = pd.read_csv(csv_path, sep = sep).drop_duplicates(subset = "ti_id").reset_index(drop=True)
    merged_df = pd.merge(df, metadata, left_on=left_on, right_on=right_on)
    merged_df.drop(columns=right_on, inplace=True)
    return merged_df

"""
This function is for tokenizing and aligning.
"""

def tokenize_and_align_labels(examples, model_checkpoint = "emanjavacas/GysBERT"):
    """ 
    Maps tokens to their labels.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=True, max_length = 512)

    labels = []
    for i, label in enumerate(examples[f"tagged"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs