from typing import Union
from dataclasses import dataclass
from utils import parse_args_to_dataclass

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

@dataclass
class Configuration:
    model_name: str
    test_path: str = "data/test_text.csv"
    submission_path: str = ""
    device: Union[str, int] = "cuda" if torch.cuda.is_available() else "cpu" 

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
