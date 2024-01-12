
def custom_model_annotation(df, model_name, aggregation_strategy = "average"):
    # Load the token classification pipeline
    pipe = pipeline("token-classification", model=model_name, aggregation_strategy=aggregation_strategy,)

    # Function to process a single sentence and extract token classifications
    def process_sentence(sentence):
        output = pipe(sentence)
        return [[entity["start"], entity["end"], entity["entity_group"]] for entity in output]

    # Apply the function to each sentence in the dataframe with a progress bar
    tqdm.pandas(desc="Processing Sentences")
    df['huggingface_labels'] = df['sentence'].progress_apply(process_sentence)
    return df

def tokenize_with_spans_including_punctuation(sentence):
    # Tokenize the sentence and keep track of the start and end indices of each word and punctuation
    words = []
    spans = []

    # Tokenize with word boundaries, including punctuation
    for match in re.finditer(r'\b\w+\b|[,.:;!?]', sentence):
        start, end = match.span()
        word = sentence[start:end]
        words.append(word)
        spans.append((start, end))

    return words, spans

def apply_labels_to_tokens_including_punctuation(sentence, labeled_spans, tag2id):
    words, word_spans = tokenize_with_spans_including_punctuation(sentence)
    labels = ['O'] * len(words)  # Initialize all labels as 'O'

    if labeled_spans == []:
        return words, labels

    for span_start, span_end, label in labeled_spans:
        for i, (start, end) in enumerate(word_spans):
            if start >= span_start and end <= span_end:
                if start == span_start:
                    labels[i] = f"B-{label}"
                else:
                    labels[i] = f"I-{label}"

    return words, labels

def convert_labels_to_numeric(labels, tag2id):
    return [tag2id[label] if label in tag2id else tag2id['O'] for label in labels]

def to_doccano_format(sentences, classifier):
    json_lst = []
    for sentence in tqdm(sentences):
        hf_output = classifier(sentence)
        if bool(hf_output) == True:
            labels = [[d["start"], d["end"], d["entity_group"]] for d in hf_output]
            data = {"text": sentence, "label": labels}
            json_lst.append(json.dumps(data))
    return json_lst


def jsonl_to_ner_format(filepath, tag2id, text_col="text", label_col="label"):
    lst = []
    with open(filepath, "r") as file:
        for line in file:
            data = json.loads(line)
            text, entities = data[text_col], data[label_col]
            doc = nlp(text)
            biluo_tags = offsets_to_biluo_tags(doc, entities)
            iob_tags = biluo_to_iob(biluo_tags)
            try:
                number_tags = [tag2id[iob] for iob in iob_tags]
                together = [[str(word) for word in doc], number_tags]
                lst.append(together)
            except:
                print(doc)
    return lst


def compute_metrics_for_initial_training(p, label_lst):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_lst[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_lst[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_metrics_without_O(p, label_lst):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Determine the index of the "O" label
    O_index = label_lst.index("O")

    # Filter out predictions and labels with -100 and "O" labels
    true_predictions = [
        [
            label_lst[p]
            for (p, l) in zip(prediction, label)
            if l != -100 and p != O_index and l != O_index
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            label_lst[l]
            for (p, l) in zip(prediction, label)
            if l != -100 and p != O_index and l != O_index
        ]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions, references=true_labels, zero_division=0
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def process_batch_llm(batch, chain):
    """
    This function takes the sentences from a batch, 
    hands them to a chain and returns results.
    """
    results = []
    for sentence in batch:
        response = chain.invoke(sentence)
        tagged = transform_output(sentence, response)
        results.append(tagged)
    return results

def align_labels_with_tokens_IOB(tokenizer, sentence, labels):
    """
    This takes a combination of sentence and labels and
    both converts the sentence to tokens and converts
    the labels to IOB format.
    """
    tokenized_input = tokenizer(sentence, return_offsets_mapping=True)
    token_labels = ["O"] * len(tokenized_input["input_ids"])  # Initialize labels as 'Outside'
    
    if labels == "[]" :
        return token_labels
    
    labels = ast.literal_eval(str(labels))
    for start, end, label in labels:
        token_start = None
        token_end = None

        # Find the token indices that correspond to the start and end of the span
        for idx, (token_start_pos, token_end_pos) in enumerate(tokenized_input.offset_mapping):
            if token_start_pos == 0 and token_end_pos == 0:
                continue
            if token_start is None and token_start_pos >= start:
                token_start = idx
            if token_end_pos <= end:
                token_end = idx

        # Assign labels (using IOB format)
        if token_start is not None and token_end is not None:
            token_labels[token_start] = f"B-{label}"
            for idx in range(token_start + 1, token_end + 1):
                token_labels[idx] = f"I-{label}"

    # Remove the label for special tokens (like [CLS], [SEP])
    token_labels = [label if offset_mapping[0] != offset_mapping[1] else "O" for label, offset_mapping in zip(token_labels, tokenized_input.offset_mapping)]
    return token_labels

def estimate_token_count(text, char_per_tok):
    """
    Estimates the numbers of tokens in a text.
    For my usecase (historic Dutch and json output)
    one token per two characters is reasonable, but this
    might be different for other usecases.
    """
    return len(str(text)) / char_per_tok

def process_batch_with_token_count(batch, chain, char_per_tok=2):
    """
    Processes a batch and returns the estimated costs.
    """
    input_token_count = 0
    output_token_count = 0
    results = []

    for sentence in batch:
        # Count input tokens
        input_tokens = estimate_token_count(sentence, char_per_tok)
        input_token_count += input_tokens

        # API call
        response = chain.invoke(sentence)

        # Count output tokens
        output_tokens = estimate_token_count(response, char_per_tok)
        output_token_count += output_tokens

        # Process response
        tagged = transform_output(sentence, response)
        results.append(tagged)

    return results, input_token_count, output_token_count


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

def align_labels_with_tokens(tokenizer, sentence, labels):
    """
    Converts the label spans to IOB-notation
    on the token level. 

    This prepares the data for training of a token
    classification model.     
    """
    tokenized_input = tokenizer(sentence, return_offsets_mapping=True)
    token_labels = ["O"] * len(tokenized_input["input_ids"])  # Initialize labels as 'Outside'

    for start, end, label in labels:
        token_start = None
        token_end = None

        # Find the token indices that correspond to the start and end of the span
        for idx, (token_start_pos, token_end_pos) in enumerate(tokenized_input.offset_mapping):
            if token_start is None and token_start_pos >= start:
                token_start = idx
            if token_end_pos <= end:
                token_end = idx

        # Assign labels (using IOB format)
        if token_start is not None and token_end is not None:
            token_labels[token_start] = f"B-{label}"
            for idx in range(token_start + 1, token_end + 1):
                token_labels[idx] = f"I-{label}"

    # Remove the label for special tokens (like [CLS], [SEP])
    token_labels = [label if offset_mapping[0] != offset_mapping[1] else "O" for label, offset_mapping in zip(token_labels, tokenized_input.offset_mapping)]
    return token_labels