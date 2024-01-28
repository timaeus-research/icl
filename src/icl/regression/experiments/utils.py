from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import infra

K=3  # Num cov components


def iter_models(model, checkpointer, verbose=False):
    for file_id in tqdm(checkpointer.file_ids, desc="Iterating over checkpoints", disable=not verbose):
        model.load_state_dict(checkpointer.load_file(file_id)["model"])
        yield model



def process_tensor(a):
    if len(a.shape) == 0 or a.shape == (1,):
        return a.item()
    return a.tolist()


def flatten_and_process(dict_, delimiter='/', prefix=""):
    flattened = {}
    for key, value in dict_.items():
        if isinstance(value, torch.Tensor):
            value = process_tensor(value)
        
        if isinstance(value, dict):
            flattened.update(
                flatten_and_process(
                    value,
                    prefix=f"{prefix}{key}{delimiter}",
                    delimiter=delimiter,
                )
            )
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, (dict, list)):
                    flattened.update(
                        flatten_and_process(
                            {str(i): v},
                            prefix=f"{prefix}{key}{delimiter}",
                            delimiter=delimiter,
                        )
                    )
                else:
                    flattened[f"{prefix}{key}{delimiter}{i}"] = v

            if isinstance(value[0], (int, float)):
                flattened[f"{prefix}{key}{delimiter}mean"] = np.mean(value)
                flattened[f"{prefix}{key}{delimiter}std"] = np.std(value)
        else:
            flattened[f"{prefix}{key}"] = value

    return flattened