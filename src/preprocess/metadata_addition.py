"""
Add metadata to a dataframe with dbnl sentences.
The metadata is connected based on the dbnl-id
of the text. 

NB: I use drop_duplicates here, this might make the metadata
incomplete on some aspects.
"""
import pandas as pd

def join_meta_csv(df, csv_path, left_on="text_id", right_on="ti_id", sep = ","):
    metadata = pd.read_csv(csv_path, sep = sep).drop_duplicates(subset = "ti_id").reset_index(drop=True)
    merged_df = pd.merge(df, metadata, left_on=left_on, right_on=right_on)
    merged_df.drop(columns=right_on, inplace=True)
    return merged_df

