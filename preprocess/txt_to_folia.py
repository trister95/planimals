"""
These function enable tokenizing 
the dbnl-texts. The output is folia.xml-files.
"""
import os
import ucto
from preproces.dbnl_xml_to_txt import extract_dbnl_id

def list_txt_files(directory):
    """ 
    Outputs all .txt.-files from a directory.
    """
    txt_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_files.append(filename)
    return txt_files

def txt_to_folia(input_dir, output_dir):
    """
    Tokenizes txt files by converting them to folia.xml-files.
    """
    configurationfile_ucto = "tokconfig-nld-historical" 
    tokenizer = ucto.Tokenizer(configurationfile_ucto, foliaoutput = True)

    for f in list_txt_files(input_dir):
        p = input_dir + "/" + f
        out_path = str(output_dir) + "/" + extract_dbnl_id(f) + ".folia.xml"
        tokenizer.tokenize(p, out_path)
    return


