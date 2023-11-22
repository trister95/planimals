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


def get_text_dbnl_(xml_file, id, output_dir):
    """
    Extracts the text from a dbnl xml-file
    and saves it as .txt-file.
    """
    xml = lxml.etree.parse(xml_file)
    check_multiple(xml)

    text = "".join(xml.find("//text").itertext())
    with open(f"{output_dir}/{id}.txt", "w") as f:
        f.write(text)
    return


def extract_text(elem):
    """Extracts and cleans text from an XML element, skipping comments."""
    if isinstance(elem, ET._Comment):
        return ""
    return clean_whitespace("".join(elem.itertext()))


def extract_notes(elem):
    """Extracts and cleans text from a note element."""
    if elem.tag == "note":
        return clean_whitespace("".join(elem.itertext()))
    return ""


def combine_text(main_text, notes, include_notes):
    """Combines main text and notes based on the include_notes option."""
    if include_notes == "end":
        return "\n".join(main_text) + "\n\nNotes:\n" + "\n".join(notes)
    elif include_notes == "only":
        return "\n".join(notes)
    else:  # Default
        return "\n".join(main_text)


def get_text_dbnl(xml_file, id, output_dir, include_notes="default"):
    """
    Extracts the text from a dbnl xml-file and saves it as .txt-file.
    Only includes text within the <text> element.
    Raises an error if multiple <text> elements are found.
    Cleans up unnecessary whitespaces.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)

    # Find the <text> elements
    text_elements = tree.findall(".//text")
    if not text_elements:
        raise ValueError("No <text> element found in the XML file.")
    if len(text_elements) > 1:
        raise ValueError("Multiple <text> elements found in the XML file.")

    # Extracting and cleaning text and notes
    main_text_parts = [
        extract_text(elem)
        for elem in text_elements[0].iter()
        if include_notes != "only"
    ]
    note_parts = [extract_notes(elem) for elem in text_elements[0].iter()]

    # Combine text based on the chosen option
    combined_text = combine_text(main_text_parts, note_parts, include_notes)

    # Save the combined text to a file
    with open(f"{output_dir}/{id}.txt", "w", encoding="utf-8") as f:
        f.write(combined_text)

    return


#


def folia_to_df(folia_path):
    """
    Converts a folia file to a df, with columns for the dbnl_id,
    the intact sentences and the sentences split up in words.
    """
    try:
        doc = folia.Document(file=folia_path)
        sentence_divided_lst = single_split_folia(doc)
        # word_divided_lst = double_split_folia(doc)
        df = pd.DataFrame(
            {"sentences": sentence_divided_lst}
        )  #'split': word_divided_lst})
        df["text_id"] = extract_dbnl_id(folia_path)
        return df
    except Exception as e:
        print(f"Error processing file {folia_path}: {e}")
        return pd.DataFrame()
