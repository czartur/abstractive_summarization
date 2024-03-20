from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from datasets import DatasetDict, Dataset, load_metric
import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass, fields
import argparse
from typing import Union

nltk.download("punkt")
rouge_score = load_metric("rouge")

@dataclass 
class Configuration:
    model_name: str
    data_path: str = "data" 
    max_target_length: int = 50
    max_input_length: Union[int, str] = "model_max_length"
    num_train_epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 5.6e-5
    weight_decay: float = 0.01

def dataclass_to_argparse(dc):
    parser = argparse.ArgumentParser()
    for dc_field in fields(dc):
        field_type = dc_field.type
        field_name = dc_field.name.replace('_', '-')
        if field_type is bool:
            parser.add_argument(
                f'--{field_name}',
                action='store_true',
                help=f'{field_name} (default: {dc_field.default})'
            )
            parser.add_argument(
                f'--no-{field_name}',
                dest=field_name,
                action='store_false'
            )
            parser.set_defaults(**{field_name: dc_field.default})
        else:
            parser.add_argument(
                f'--{field_name}',
                type=field_type,
                default=dc_field.default,
                help=f'{field_name} (default: {dc_field.default})'
            )
    return parser

def parse_args_to_dataclass(dc_cls):
    parser = dataclass_to_argparse(dc_cls)
    args = parser.parse_args()
    return dc_cls(**vars(args))

def prepare_data(tokenizer, config):
    # csv --> pandas --> dataset
    datasets = DatasetDict()
    for split in ["train", "validation"]:
        datasets[split] = Dataset.from_pandas(pd.read_csv(os.path.join(config.data_path, f"{split}.csv")))

    # tokenize
    def encode(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=config.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            examples["titles"], max_length=config.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_datasets = datasets.map(encode, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        datasets["train"].column_names
    )
    return tokenized_datasets

def compute_metrics(eval_pred): # from H.F. tutorial
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

if __name__ == "__main__":
    config = parse_args_to_dataclass(Configuration)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    tokenized_datasets = prepare_data(tokenizer, config)
    
    
    num_train_epochs = config.num_train_epochs
    batch_size = config.batch_size
    logging_steps = len(tokenized_datasets["train"]) // batch_size
    file_prefix = config.model_name.split("/")[-1]

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{file_prefix}-dc",
        evaluation_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=config.weight_decay,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=True,
        save_strategy="no"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()
