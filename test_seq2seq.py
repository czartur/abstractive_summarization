from typing import Union, get_origin
from dataclasses import dataclass, fields
import argparse
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

@dataclass
class Configuration:
    model_name: str
    test_path: str = "data/test_text.csv"
    submission_path: str = ""
    device: Union[str, int] = "cuda" if torch.cuda.is_available() else "cpu" 


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

def from_pipeline(model, tokenizer, config):
    summarizer = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        truncation=True
    )

    test_df = pd.read_csv(config.test_path)

    summarized = summarizer(test_df["text"].tolist())
    summaries = [s['summary_text'] for s in summarized]
    return summaries

if __name__ == "__main__":
    config = parse_args_to_dataclass(Configuration)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name) 
    
    summaries = from_pipeline(model, tokenizer, config)
    

    final_df = pd.DataFrame({
        "ID": range(len(summaries)),
        "titles": summaries,  
    })
    
    submission_path = config.submission_path if config.submission_path else config.model_name.split('/')[-1] + ".csv"
    final_df.to_csv(submission_path, index=False)
