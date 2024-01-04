from pathlib import Path

import devinfra
import torch
from devinfra.utils.iterables import flatten_dict, rm_none_vals
from tqdm import tqdm

K=3  # Num cov components


def iter_models(model, checkpointer, verbose=False):
    for file_id in tqdm(checkpointer.file_ids, desc="Iterating over checkpoints", disable=not verbose):
        model.load_state_dict(checkpointer.load_file(file_id)["model"])
        yield model



def process_tensor(a):
    if len(a.shape) == 0 or a.shape == (1,):
        return a.item()
    return a.tolist()


def flatten_and_process(dict_):
    return flatten_dict({
        k: process_tensor(v) if isinstance(v, torch.Tensor)  else v
        for k, v in dict_.items()
    }, flatten_lists=True)
