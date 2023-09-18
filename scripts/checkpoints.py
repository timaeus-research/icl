"""
Just use the default aws s3 command line tool to move/copy files over.

E.g.: `aws s3 cp s3://devinterp/checkpoints/icl/ s3://devinterp/backups/icl/ --recursive`

"""

import io
import os
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import boto3
import torch
import typer
from devinterp.ops.storage import (  # Import the base class and IDType
    BaseStorageProvider, IDType, S3StorageProvider, create_storage_provider,
    int_id_to_key, key_to_int_id)
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer()


def _migrate(
    client,
    bucket: str,
    prefix: str,
    target_device: str,
):
    """Change the device of the checkpoints."""
    file_ids = client.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']

    if len(file_ids) == 1000:
        warnings.warn("There are at least 1000 files in the directory. Only changing the first 1000.")        
    
    raise NotImplementedError("TODO: implement this")


@app.command()
def migrate(
    bucket: str = typer.Option(..., help="S3 bucket name"),
    prefix: str = typer.Option(..., help="Prefix"),
    device: str = typer.Option("cpu", help="Target device"),
):
    """Change the device of the checkpoints."""
    client = boto3.client('s3')
    _migrate(client, bucket, prefix target_device)


if __name__ == "__main__":
    app()
