import os
from typing import List, Union
from dataclasses import dataclass
from utils import parse_args_to_dataclass

import pandas as pd
from summarizer.sbert import SBertSummarizer
from tqdm import tqdm

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
