# INF582

This repository contains scripts for training/fine-tuning, and testing summarizations models. We used them for the [INF582 News Articles Title Generation Kaggle competition](https://www.kaggle.com/competitions/inf582-news-articles-title-generation).

📝 Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
   - [Extractive Summarization](#extractive-summarization)
   - [Training](#training)
   - [Testing](#testing)
3. [Project Structure](#project-structure)

## Installation
Before using the scripts install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```
Additionally, make sure to log in to Hugging Face:
```bash
huggingface-cli login
```
This step is necessary for uploading models in the Hugging Face hub.

## Usage
### Training
To train a model for summarization, use the `train_seq2seq.py` script. This script allows you to fine-tune a pre-trained model available in the Hugging Face model hub on your specific dataset.

Example usage:
```bash
python train_seq2seq.py \
    --model-name facebook/bart-base \       ## Hugging Face model name
    --input-dir data \                      ## Path to the datasets dir (containing a 'train.csv' and 'validation.csv')
    --repo-id bart-finetuned \              ## Choose a name for the fine-tuned model repository
    --num-train-epochs 4 \                  ## Number of training epochs
    --batch-size 4 \                        ## Batch size for training
    --learning-rate 5.6e-5 \                ## Learning rate for optimization
    --weight-decay 0.01 \                   ## Weight decay for regularization
    --max-target-length 50 \                ## Maximum length of the titles
    --max-input-length model_max_length     ## Maximum length of the documents
```

### Testing
Once you have trained a model or have access to a pre-trained model, you can use the `test_seq2seq.py` script to generate summaries for a dataset.

Example usage:
```bash
python3 test_seq2seq.py \
    --model-name czartur/bart-finetuned \    ## Hugging Face model name
    --test-path data/test_text.csv \         ## Path to the test dataset
    --submission-path "submission.csv" \     ## Path for the output file
    --device "cuda"                          ## Device for inference (default: "cuda" if available, else "cpu")

```

### Extractive Summarization
The `extsum.py` script allows you to perform extractive summarization using the SbertSummarizer. This method extracts key sentences from the input text to form a summary.

Example usage:
```bash
python3 extsum.py \
    --model-name "paraphrase-MiniLM-L6-v2" \ ## Model name for text summarization
    --input-path "data" \                    ## Path to the input data
    --output-path "ext_data" \               ## Path for the output extracted summaries
    --num-sentences None                     ## Number of sents per doc to extract (if None, model selects auto)
```

## Project Structure
```markdown
.
├── data                                     # dataset
│   ├── train.csv
│   ├── validation.csv
│   └── test_text.csv
├── requirements.txt                         # pip required packages
├── utils.py                                 # utility functions
├── test_seq2seq.py                          # script for testing a model (generate output)
├── train_seq2seq.py                         # script for fine tuning a model
└── extsum.py                                # script for extractive summarize the dataset
```
