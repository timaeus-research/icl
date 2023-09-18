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
import tqdm
import typer
from devinterp.ops.storage import (  # Import the base class and IDType
    BaseStorageProvider, IDType, S3StorageProvider, create_storage_provider,
    int_id_to_key, key_to_int_id)
from dotenv import load_dotenv

from icl.utils import to

load_dotenv()
app = typer.Typer()

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
):
    """Change the device of the checkpoints."""
    file_keys = get_all_file_keys(client, bucket, prefix, max_keys=max_keys)
    
    for file_key in tqdm.tqdm(file_keys, desc="Migrating files", unit="files"):
        # Download the file
        obj = client.get_object(Bucket=bucket, Key=file_key)
        body = obj['Body'].read()
        stream = io.BytesIO(body)
        checkpoint = torch.load(stream)
        pp(checkpoint)  # TODO: Remove
        checkpoint = to(checkpoint, device)
        pp(checkpoint)  # TODO: Remove
        
        return # TODO: Remove
        
        # Upload the checkpoint
        client.put_object(Bucket=bucket, Key=file_key, Body=io.BytesIO(body))


@app.command()
def migrate(
    bucket: str = typer.Argument(..., help="S3 bucket name"),
    prefix: str = typer.Argument("/", help="Prefix"),
    to: str = typer.Option("cpu", help="Target device"),
    max_keys: Optional[int] = typer.Option(None, help="Maximum number of keys to migrate"),
):
    """Change the device of the checkpoints."""
    client = boto3.client('s3')
    _migrate(client, bucket, prefix, to, max_keys=max_keys)


if __name__ == "__main__":
    app()
