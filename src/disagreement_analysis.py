from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

def compare_annotations_as_strings(df, llm_col, ner_col):
    # Convert list to string
    # Compare and create the disagreement column
    df['disagreement'] = df.apply(lambda row: str(row[llm_col]) != str(row[ner_col]), axis=1)
    return df

def list_to_string(lst):
    return str(lst)

def process_sentence(sentence, tokenizer, pipe, max_length=512, overlap=50):
    tokens = tokenizer.tokenize(sentence)
    adjusted_max_length = max_length -2

    if len(tokens) <= adjusted_max_length:
        output = pipe(sentence)
        return [[entity["start"], entity["end"], entity["entity_group"]] for entity in output]

    chunks = [tokens[i:i + adjusted_max_length] for i in range(0, len(tokens), adjusted_max_length - overlap)]
    all_entities = []
    current_position = 0

    for chunk in chunks:
        chunk_str = tokenizer.convert_tokens_to_string(chunk)
        output = pipe(chunk_str)

        adjusted_entities = []
        for entity in output:
            adjusted_entity = {
                "start": entity["start"] + current_position,
                "end": entity["end"] + current_position,
                "entity_group": entity["entity_group"]
            }
            adjusted_entities.append(adjusted_entity)   
        all_entities.extend(adjusted_entities)     

        current_position += len(chunk_str) - (overlap * len(" "))
        #theoretically 1 token can get different labels now, but I don't think it matters for this purpose
    return all_entities


def custom_model_annotation(df, model_name, max_length=512, overlap=50, aggregation_strategy="average"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("token-classification", model=model_name, aggregation_strategy=aggregation_strategy)

    tqdm.pandas(desc="Processing Sentences")
    df['huggingface_labels'] = df['sentence'].progress_apply(lambda x: process_sentence(x, tokenizer, pipe, max_length, overlap))
    return df

