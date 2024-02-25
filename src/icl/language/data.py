import json
import os
import pickle
from typing import Union

import boto3
import datasets
import torch
import tqdm
from huggingface_hub import HfApi, HfFolder, create_repo
from transformer_lens.utils import AutoTokenizer, tokenize_and_concatenate

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



def upload_dataset_to_hub(dataset, dataset_name: str, organization: str = None, force_save=False):
    # Define the repository name and whether it belongs to an organization or a user
    repo_name = dataset_name if organization is None else f"{organization}/{dataset_name}"
    
    # Login to Hugging Face (make sure you've logged in via CLI)
    api = HfApi()
    token = HfFolder.get_token()
    assert token, "You must be logged in to Hugging Face Hub"
    
    # Create a new dataset repository on Hugging Face Hub
    repo = create_repo(repo_name, token=token, repo_type="dataset", exist_ok=True)
    print(f"Using dataset repository: {repo}")
    
    # Save dataset to a local directory
    dataset_path = f"../tmp/{dataset_name}"

    if not os.path.exists(dataset_path) or force_save:
        print(f"Saving dataset '{dataset_name}' to {dataset_path}...")
        dataset.save_to_disk(dataset_path)
    else:
        print(f"Dataset '{dataset_name}' already exists in the local directory.")
    
    # Initialize a git repository in this directory, add, commit, and push files to Hugging Face Hub
    api.upload_folder(
        token=token,
        repo_id=repo_name,
        folder_path=dataset_path,
        repo_type="dataset"
    )
    
    print(f"Dataset '{dataset_name}' uploaded to Hugging Face Hub: https://huggingface.co/datasets/{repo_name}")

    # Clean up the local directory
    os.system(f"rm -rf {dataset_path}")


def get_tokenized_dataset(original_dataset_name, tokenizer: Union[AutoTokenizer, str], max_length=1024, force_reprocess=False):
    # Attempt to load the tokenized dataset
    tokenized_dataset_name = f"timaeus/{''.join(original_dataset_name.split('/')[1:])}-tokens"

    if force_reprocess:
        print(f"Force reprocessing is enabled, reprocessing {tokenized_dataset_name}...")
    else:
        try:
            tokenized_dataset = datasets.load_dataset(tokenized_dataset_name, split='train')
            print(f"Tokenized dataset {tokenized_dataset_name} already exists.")
            return tokenized_dataset
        
        except FileNotFoundError:
            print(f"Tokenized dataset {tokenized_dataset_name} not found, processing...")
        except Exception as e:
            print(f"Error loading tokenized dataset {tokenized_dataset_name}, processing...")
            print(e)

    # Load the original dataset
    dataset = datasets.load_dataset(original_dataset_name, split='train')
    
    # Tokenization and transformation logic here
    # This is a placeholder for the actual tokenization and processing
    tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer

    print(f"Tokenizing {original_dataset_name} with {tokenizer.name_or_path}...")

    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=max_length,
        column_name='contents',
        add_bos_token=True,
        num_proc=12
    )
    
    # Upload the tokenized dataset to Hugging Face (implement the actual upload logic)
    # This is a placeholder function call
    upload_dataset_to_hub(tokenized_dataset, tokenized_dataset_name)
    
    # Clear the cache for the original dataset
    # Be cautious with this step to avoid unintentional data loss   
    dataset.cleanup_cache_files()
    del dataset

    return tokenized_dataset


def get_loader(dataset, batch_size=100, num_workers=0, shuffle=True, pin_memory=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )
    return dataloader
   