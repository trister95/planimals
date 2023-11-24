import pandas as pd
import langchain
import openai
import re
import os
import random
from transformers import AutoTokenizer
from environs import Env
from tqdm import tqdm
from utils import list_extension_files
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.output_parsers.json import SimpleJsonOutputParser


def find_indices(sentence, word):
    """
    Find the span indices that match a word or phrase.
    """
    start_index = sentence.find(word)
    end_index = start_index + len(word)
    return [start_index, end_index]


def has_multiple_occurrences(sentence, model_output):
    """
    Checks if a word has multiple occurences in
    a sentence.
    """
    for category in model_output:
        for word in model_output[category]:
            if sentence.count(word) > 1:
                return True
    return False


def transform_output(sentence, model_output):
    """
    Transforms the output of the LLM to a format that can later
    be used for converting to IOB-notation.

    If a word has multiple occurences. The sentence is flagged
    so it can be checked manually.
    """
    labels = []
    flagged = False

    if has_multiple_occurrences(sentence, model_output):
        return {"Sentence": sentence, "Labels": [], "Flagged": True}

    for category in model_output.keys():
        for word in model_output[category]:
            indices = find_indices(sentence, word)
            if indices:
                labels.append(indices + [category])  # 'animal' or 'plant'

    return {"Sentence": sentence, "Labels": labels, "Flagged": flagged}


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

def align_labels_with_tokens_IOB(tokenizer, sentence, labels):
    """
    This takes a combination of sentence and labels and
    both converts the sentence to tokens and converts
    the labels to IOB format.
    """
    tokenized_input = tokenizer(sentence, return_offsets_mapping=True)
    token_labels = ["O"] * len(tokenized_input["input_ids"])  # Initialize labels as 'Outside'

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