import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import boto3
import numpy as np
import torch
from torch import nn

from huggingface_hub import HfApi
from transformer_lens import HookedTransformer

from icl.constants import DEVICE
from icl.language.model import get_model_cfg


AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
HF_API = HfApi(
    endpoint="https://huggingface.co",
    token=None,
)

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

def load_hf_checkpoint(step, n_layers=2):
  model_cfgs = {}
  model_cfgs[1] = get_model_cfg(num_layers=1)
  model_cfgs[2] = get_model_cfg(num_layers=2)
  repo_id = f'oknMswoztTPaAVreBrWy/L{n_layers}'
  checkpoint_name = f'checkpoint_{step:0>7d}.pth'
  model_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=f'checkpoints/{checkpoint_name}')
  state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
  checkpoint = HookedTransformer(model_cfgs[n_layers])
  checkpoint.load_state_dict(state_dict)
  return checkpoint