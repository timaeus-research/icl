"""
Just use the default aws s3 command line tool to move/copy files over.

E.g.: `aws s3 cp s3://devinterp/checkpoints/icl/ s3://devinterp/backups/icl/ --recursive`

"""
import io
import os
import pickle
import warnings
from pathlib import Path
from pprint import pp
from typing import Any, Callable, List, Optional, Union

import boto3
import torch
import torch_xla
import tqdm
import typer
from devinfra.io.storage import (  # Import the base class and IDType
    BaseStorageProvider, IDType, S3StorageProvider, create_storage_provider,
    int_id_to_key, key_to_int_id)
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer()

try


def move_to_(obj, device = "cpu"):
    """
    Moves the given object to the given device.
    """

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            move_to_(item, device)
    elif isinstance(obj, dict):
       for value in obj.values():
           move_to_(value, device)
    elif hasattr(obj, "to"):
        obj.to(device)


def get_all_file_keys(client, bucket: str, prefix: str, max_keys: Optional[int] = None):
    # Initialize an empty list to hold file ids
    files = []

    # Initialize variables for pagination
    next_token = ''
    pbar = tqdm.tqdm(desc="Listing files", unit="files")

    while True:
        if next_token:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=next_token)
        else:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        # Check if 'Contents' are present
        if 'Contents' in response:
            files.extend(response['Contents'])
        
        # Check for 'NextContinuationToken' for further pagination
        if 'NextContinuationToken' in response and (not max_keys or len(files) < max_keys):
            next_token = response['NextContinuationToken']
        else:
            break

        pbar.update(len(response['Contents']))

    return [file["Key"] for file in files]


def _migrate(
    client,
    bucket: str,
    prefix: str,
    device: str,
    max_keys: Optional[int] = None,
    resume: Optional[int] = None
):
    """Change the device of the checkpoints."""
    file_keys = sorted(get_all_file_keys(client, bucket, prefix, max_keys=max_keys))

    start_idx = resume or 0
    for i, file_key in enumerate(tqdm.tqdm(file_keys[start_idx:], desc="Migrating files", unit="files", initial=start_idx, total=len(file_keys))):
        # Download the file
        obj = client.get_object(Bucket=bucket, Key=file_key)
        body = obj['Body'].read()
        stream = io.BytesIO(body)
        checkpoint = torch.load(stream)
        move_to_(checkpoint, device)
        assert checkpoint["model"]["token_sequence_transformer.token_embedding.weight"].device == torch.device(device)
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        # Upload the checkpoint

        client.put_object(Bucket=bucket, Key=file_key, Body=stream.getvalue())


@app.command()
def migrate(
    bucket: str = typer.Argument(..., help="S3 bucket name"),
    prefix: str = typer.Argument("/", help="Prefix"),
    to: str = typer.Option("cpu", help="Target device"),
    max_keys: Optional[int] = typer.Option(None, help="Maximum number of keys to migrate"),
    resume: Optional[int]= typer.Option(None, help="Index of the file to resume migrating from.")
):
    """Change the device of the checkpoints."""
    client = boto3.client('s3')
    _migrate(client, bucket, prefix, to, max_keys=max_keys, resume=resume)


if __name__ == "__main__":
    app()
