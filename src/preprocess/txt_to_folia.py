"""
These function enable tokenizing 
the dbnl-texts. The output is folia.xml-files.
"""
import os
import ucto
import tqdm
from preprocess.dbnl_xml_to_txt import extract_dbnl_id
from utils.list_extension_files import list_extension_files

def txt_to_folia(input_dir, output_dir):
    """
    Tokenizes txt files by converting them to folia.xml files.
    """
    configurationfile_ucto = "tokconfig-nld-historical" 
    tokenizer = ucto.Tokenizer(configurationfile_ucto, foliaoutput=True)

    for f in tqdm.tqdm(list_extension_files(input_dir, "txt"), desc="Looping through text files"):
        input_path = os.path.join(input_dir, f)
        output_filename = f"{extract_dbnl_id(f)}.folia.xml"
        out_path = os.path.join(output_dir, output_filename)
        
        if not os.path.exists(out_path):
            tokenizer.tokenize(input_path, out_path)
