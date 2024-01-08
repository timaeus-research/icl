import os
import pickle
from typing import Dict, List, Union

import boto3
import numpy as np
import torch
from torch import nn

from icl.constants import DEVICE


def translate_int_to_str(token_ints: List[int], model: nn.Module):
    t = torch.tensor([token_ints], device=DEVICE)
    tokens = model.to_str_tokens(t)

    if isinstance(token_ints, int):
        return tokens[0]
    return tokens


def save_to_bucket(filename, data: Union[Dict, np.ndarray]):
    client = boto3.client('s3')

    with open(f'/tmp/{filename}.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open(f'/tmp/{filename}.pkl', 'rb') as f:
        client.upload_fileobj(f, 'devinterp', f'other/language/{filename}.pkl')

    os.remove(f'/tmp/{filename}.pkl')