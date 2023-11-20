"""
These functions extract the actual literary texts
from a directory with dbnl-xml files.
"""
import os
import regex as re
import lxml.etree
from ..utils import extract_dbnl_id

def list_xml_files(directory):
    """ 
    Lists all the xml files in a directory.
    """
    xml_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_files.append(filename)
    return xml_files

def check_multiple(tree):
    """ 
    Makes sure there aren't multiple instantiations of //text
    (code wouldn't work properly in that scenario).    
    """
    count = tree.xpath('count(//text)')
    if count > 1:
        print("Warning: multiple text elements in one doc. This code doesn't work for this situation!")
    return

def get_text_dbnl(xml_file, id, output_dir):
    """ 
    Extracts the text from a dbnl xml-file
    and saves it as .txt-file.
    """
    xml = lxml.etree.parse(xml_file) 
    check_multiple(xml)

    text = ''.join(xml.find("//text").itertext())
    with open(f'{output_dir}/{id}.txt', 'w') as f:
        f.write(text)
    return

def dbnl_to_txt(input_dir, output_dir):
    """ 
    Puts the pieces of the pipeline together.
    """
    xml_files = list_xml_files(input_dir)
    for f in xml_files:
        id = extract_dbnl_id(f)
        p = input_dir + "/" + f
        try:
            get_text_dbnl(p, id, output_dir)
        except:
            print(f"file {id} could not be parsed")
    return