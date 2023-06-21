""" 
These functions adds declarations for html character entities
to the xml files. In this way, xml files with html character
entities can be read by a xml parser.
"""

import os
import html.entities

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
  Provide it with the end of the old declaration (make sure this only pops up once in
  the document!) and provide a dictionary. 

  Note that this function is written for conversion of html character entities
  to numerical representations. For other use, some tweaks might be needed.
  """
  new_declaration = ""
  
  for item in html_decimal_dict.items():
    new_str = f'<!ENTITY {item[0]} "&#{item[1]};">\n'
    new_declaration += new_str

  return  recognition_str+ '" [' + new_declaration + ']'


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
      new_doctype = doctype.replace(end, addition)

      if new_doctype not in xml_text:
        xml_text = xml_text[:start] + new_doctype + xml_text[end:]
        with open(os.path.join(xml_dir, f_name), 'w', encoding = 'utf-8') as f:
          f.write(xml_text)
    return