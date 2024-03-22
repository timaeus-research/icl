import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import boto3
import numpy as np
import torch
from torch import nn

from icl.constants import DEVICE

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def translate_int_to_str(token_ints: List[int], model: nn.Module):
    t = torch.tensor([token_ints], device=DEVICE)
    tokens = model.to_str_tokens(t)

    if isinstance(token_ints, int):
        return tokens[0]
    return tokens


def save_to_bucket(filename: Union[str, Path], data: Union[Dict, np.ndarray]):
    client = boto3.client('s3')

    filename = str(filename)

    if "/" in filename:
        filename = filename.split("/")[-1]
    if ".pkl" not in filename:
        filename += ".pkl"

    with open(f'/tmp/{filename}', 'wb') as f:
        pickle.dump(data, f)

    with open(f'/tmp/{filename}', 'rb') as f:
        client.upload_fileobj(f, AWS_BUCKET_NAME, f'other/language/{filename}')

    os.remove(f'/tmp/{filename}')


