import json
import asyncio
from tqdm import tqdm
import re
from aiolimiter import AsyncLimiter
from typing import List, Tuple, Union
from pydantic import BaseModel, Field


class Plants_and_animals(BaseModel):
    """ A representation of all the plants and animals present in the sentence"""
    sentence: str = Field(default = None, description = "The sentence itself")
    plants: list[str] = Field(description="The plants present in the sentence")
    animals: list[str] = Field(description= "The animals present in the sentence")


class Annotations(BaseModel):
    sentence: str
    gpt_labels: List[Tuple[Union[int, None], Union[int, None], Union[str, None]]] = []
    warning: bool = False
    log: str = ""

    @classmethod
    def create_from_plants_and_animals(cls, data: Plants_and_animals):
        instance = cls(sentence=data.sentence)
        labels = []
        warnings = []

        for entity_type, entity_list in [("plants", data.plants), ("animals", data.animals)]:
            for entity in entity_list:
                occurrences = [m.start() for m in re.finditer(entity, data.sentence)]
                
                if len(occurrences) == 1:
                    start = occurrences[0]
                    end = start + len(entity)
                    labels.append((start, end, entity_type))
                elif len(occurrences) > 1:
                    warnings.append(f"Multiple occurrences of '{entity}' ({entity_type})")
                else:
                    warnings.append(f"'{entity}' ({entity_type}) not found")

        if warnings:
            instance.warning = True
            instance.log = "; ".join(warnings)
        else:
            instance.gpt_labels = labels
        return instance
    
    def to_dict(self):
        return {
            "sentence": self.sentence,
            "gpt_labels": self.gpt_labels,
            "warning": self.warning,
            "log": self.log,
        }

async def process_sentence_async(sentence, chain, limiter):
    async with limiter:
        try:
            model_output = await chain.ainvoke({"sentence": sentence})
            model_output.sentence = sentence
            return Annotations.create_from_plants_and_animals(model_output)
        except Exception as e:
            # Handle any exception
            return Annotations(
                sentence=sentence,
                gpt_labels = [],
                warning=True,
                log=f"Exception: {e}",
            )

async def process_batch_llm_async(batch, chain, limiter):
    return await asyncio.gather(*[process_sentence_async(sentence, chain, limiter) for sentence in batch])


async def process_batches(batches, chain, requests_per_minute=900):
    limiter = AsyncLimiter(requests_per_minute)
    results = []
    # Create a tqdm progress bar
    with tqdm(total=len(batches), desc="Processing Batches") as pbar:
        for batch in batches:
            batch_results = await process_batch_llm_async(batch, chain, limiter)
            results.extend(batch_results)
            pbar.update(1)  # Update the progress bar after each batch
    return results

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

