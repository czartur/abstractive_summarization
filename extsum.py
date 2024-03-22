from typing import List, Union, get_origin
from dataclasses import dataclass, fields
import argparse 
import os

from summarizer.sbert import SBertSummarizer
from tqdm import tqdm
import pandas as pd
# import torch



@dataclass
class Configuration:
    num_sentences: Union[None, int] = None # if None, model selects auto
    model_name: str = "paraphrase-MiniLM-L6-v2"
    input_path: str = "data"
    output_path: str = "ext_data"

def extractive_summarization(docs: List[str], model, num_sentences: int) -> List[str]:
    ext_docs = []
    for doc in tqdm(docs):
        ext_docs.append(model(doc, num_sentences=num_sentences))
    return ext_docs

def dataclass_to_argparse(dc):
    parser = argparse.ArgumentParser()
    for dc_field in fields(dc):
        field_type = dc_field.type
        field_name = dc_field.name.replace('_', '-')
        field_default = dc_field.default
        if field_type is bool:
            parser.add_argument(
                f'--{field_name}',
                action='store_true',
                help=f'{field_name} (default: {field_default})'
            )
            parser.add_argument(
                f'--no-{field_name}',
                dest=field_name,
                action='store_false'
            )
            parser.set_defaults(**{field_name: field_default})
        elif get_origin(field_type) == Union:
            field_types = field_type.__args__
            type_lambda = lambda x: next((t(x) for t in field_types if isinstance(x, t)), None)
            parser.add_argument(
                f'--{field_name}',
                type=type_lambda,
                default=field_default,
                help=f'{field_name} (default: {field_default})'
            )
        else:
            parser.add_argument(
                f'--{field_name}',
                type=field_type,
                default=field_default,
                help=f'{field_name} (default: {field_default})'
            )
    return parser

def parse_args_to_dataclass(dc_cls):
    parser = dataclass_to_argparse(dc_cls)
    args = parser.parse_args()
    return dc_cls(**vars(args))

if __name__ == "__main__":
    config = parse_args_to_dataclass(Configuration)
    
    # load summarizer model
    model = SBertSummarizer(config.model_name)

    # create output_path if it does not exist  
    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # assert that input_path is a directory and that it exists
    assert os.path.isdir(config.input_path)
    for file in os.listdir(config.input_path):
        if not file.endswith(".csv"): continue
        file_path = os.path.join(config.input_path, file) 
        df = pd.read_csv(file_path)
        if not 'text' in df.columns: continue

        # perform extractive summarization        
        ext_sums = extractive_summarization(df["text"].tolist(), model, num_sentences=config.num_sentences)

        # create new dataframe with extracted data as text
        ext_df = df.copy() 
        ext_df["text"] = ext_sums
        # write to file
        ext_df.to_csv(os.path.join(config.output_path, file), index=False)