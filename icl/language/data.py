import json
import os
import pickle

import boto3
import datasets
import tqdm

from icl.constants import BIGRAMS_FILEPATH, DATA, LANGUAGE_FILEPATH
from icl.monitoring import stdlogger


def download_dataset(file_path, download_path):
    dataset = datasets.load_dataset(
        download_path,
        split=None,   
    )
    dataset.to_json(file_path)


def gen_from_jsonl(file_path, num_lines=5_000_000, start=0, verbose=True):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == start:
                break

        for i, line in tqdm.tqdm(enumerate(file), total=num_lines, desc=f'Loading dataset', disable=not verbose):
            if i >= start + num_lines:
                break

            yield json.loads(line)
            

def gen_samples(model, file_path=LANGUAGE_FILEPATH, num_lines=5_000_000, start=0, verbose=True):
    print(f"Loading {file_path}...")
    if not os.path.exists(file_path):
        stdlogger.info("Downloading dataset to %s...", file_path)
        download_dataset(file_path, 'georgeyw/dsir-pile-5m')
        stdlogger.info("...done")
    
    for row in gen_from_jsonl(file_path, num_lines, start, verbose):
        contents = row['contents']
        yield model.tokenizer(contents)['input_ids']
   