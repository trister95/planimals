import pandas as pd
import json
import asyncio
from tqdm import tqdm
import re
import ast
from aiolimiter import AsyncLimiter

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

    If a word has multiple occurrences, the sentence is flagged
    so it can be checked manually.
    """
    labels = []
    flagged = False

    # Check for None or unexpected data types
    if model_output is None or not isinstance(model_output, dict):
        return {"sentence": sentence, "labels": [], "flagged": True}

    try:
        # Proceed if model_output is a dictionary as expected
        if has_multiple_occurrences(sentence, model_output):
            return {"sentence": sentence, "labels": [], "flagged": True}

        for category in model_output.keys():
            for word in model_output[category]:
                indices = find_indices(sentence, word)
                if indices:
                    labels.append(indices + [category])  # 'animal' or 'plant'

        return {"sentence": sentence, "label": labels, "flagged": flagged}

    except Exception as e:
        # Handle any other unexpected errors
        print(f"Error while transforming output: {e}")
        return {"sentence": sentence, "labels": [], "flagged": True}

def dataframe_column_to_jsonl(df, column_name, output_file):
    """
    Convert a specified column of a DataFrame into a JSONL file.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to convert.
    output_file (str): The name of the output JSONL file.
    """
    with open(output_file, 'w') as file:
        for item in df[column_name]:
            # Each item in the column is a JSON object
            json_record = json.dumps({"text": item, "label": [[]]})
            file.write(json_record + '\n')
    print(f"File '{output_file}' created successfully.")


def update_dataframe_with_annotations(original_df, jsonl_file, sentence_column, new_label_column, flagged_column):
    """
    Update the original DataFrame with a few annotations from a JSONL file.

    Parameters:
    original_df (pandas.DataFrame): The original DataFrame with a large number of sentences.
    jsonl_file (str): The file path of the annotated JSONL file with a few sentences.
    sentence_column (str): The column name of the sentences in the original DataFrame.
    new_label_column (str): The column name for the new labels to be added or updated.
    flagged_column (str): The column name for the 'flagged' status to be updated.
    """
    try:
        # Read the annotated JSONL file
        with open(jsonl_file, 'r') as file:
            for line in file:
                annotation = json.loads(line)
                text = annotation['text']  #or other key
                labels = annotation['label']  # or other key

                # Find the matching sentence in the original DataFrame and update
                match_index = original_df[original_df[sentence_column] == text].index
                if not match_index.empty:
                    try:
                        [unpacked_index] = match_index 
                        original_df.at[unpacked_index, new_label_column] = labels  # update label
                        original_df.at[unpacked_index, flagged_column] = False     # set flagged to False
                    except ValueError:
                        print(f"Multiple matches found for sentence: '{text}'. Update skipped.")

    except FileNotFoundError:
        print(f"Error: The file '{jsonl_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{jsonl_file}' does not contain valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return original_df

async def process_sentence_async(sentence, chain, limiter):
    async with limiter:
        model_output = await chain.ainvoke(input=sentence)
    return transform_output(sentence, model_output)


async def process_batch_llm_async(batch, chain, limiter):
    return await asyncio.gather(*[process_sentence_async(sentence, chain, limiter) for sentence in batch])


async def process_all_batches(batches, chain, requests_per_minute=300):
    limiter = AsyncLimiter(requests_per_minute)
    results = []
    # Create a tqdm progress bar
    with tqdm(total=len(batches), desc="Processing Batches") as pbar:
        for batch in batches:
            batch_results = await process_batch_llm_async(batch, chain, limiter)
            results.extend(batch_results)
            pbar.update(1)  # Update the progress bar after each batch
    return results

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

def process_row(row,tag2id):
    sentence = row['sentence']
    labeled_spans = row['label']

    labeled_spans = ast.literal_eval(str(labeled_spans))

    # Apply your functions
    words, labels = apply_labels_to_tokens_including_punctuation(sentence, labeled_spans, tag2id)
    numeric_labels = convert_labels_to_numeric(labels, tag2id)

    return words, numeric_labels
