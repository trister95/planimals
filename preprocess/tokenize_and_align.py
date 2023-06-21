from transformers import AutoTokenizer

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