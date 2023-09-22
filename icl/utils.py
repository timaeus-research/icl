import hashlib
import json
import random
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, TypeVar, Union

import numpy as np
import torch
import os
import pandas as pd
from devinterp.utils import flatten_dict


def hash_dict(d: dict):
    sorted_dict_str = json.dumps(d, sort_keys=True)
    m = hashlib.sha256()
    m.update(sorted_dict_str.encode('utf-8'))
    return m.hexdigest()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.cuda.manual_seed_all(seed)
    except AttributeError:
        warnings.info("CUDA not available; failed to seed")

def directory_creator(directory, new_subdir):
    # Creates a new directory if it doesn't already exist
    
    new_directory = directory + "/" + new_subdir
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    return new_directory

def open_or_create_csv(filename, headers=None):
    '''
    Open a CSV file if it exists. If it doesn't exist, create it.

    :param filename: The name/path of the CSV file
    :param headers: A list of headers to write to the new CSV, if it's being created
    :return: A DataFrame with the CSV content or an empty DataFrame with specified headers
    '''

    # Check if the file exists
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=headers)
        df.to_csv(filename, index=False)

    return df

def get_device(obj: Any):
    """Get the device of a tensor, dict of tensors, list of tensors, etc. 
    Assumes all tensors are on the same device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, dict):
        return get_device(next(iter(obj.values())))
    elif isinstance(obj, (list, tuple, set)):
        return get_device(obj[0])
    else:
        return "cpu"


def to(obj: Any, device: Union[str, torch.device]):
    """
    Moves a tensor, dict of tensors, list of tensors, etc. to the given device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)((to(v, device) for v in obj))
    else:
        return obj


@contextmanager
def temp_to(d: Dict[str, Any], device: str):
    """
    Temporarily moves a tensor, dict of tensors, list of tensors, etc. to the
    given device. Restores the original device when the context manager exits.
    """
    original_device = get_device(d)

    to(d, device)
    yield
    to(d, original_device)



def unflatten_dict(d, delimiter="."):
    """Unflatten a dictionary where nested keys are separated by sep"""
    out_dict = {}
    for key, value in d.items():
        keys = key.split(delimiter)
        temp = out_dict
        for k in keys[:-1]:
            temp = temp.setdefault(k, {})
        temp[keys[-1]] = value
    return out_dict


T = TypeVar("T")


def rm_none_vals(obj: T) -> T:
    """
    Recursively remove None values from a dictionary or list.
    """
    if isinstance(obj, dict):
        return {k: rm_none_vals(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [rm_none_vals(v) for v in obj if v is not None]

    return obj  # May be None!


def filter_objs(objs: List[T], **filters: Any) -> Generator[T, None, None]:
    """
    Filter a list of objects or dictionaries based on nested filters.

    Parameters:
        objs (List[T]): List of dictionaries or objects.
        filters (Any): Nested key-value pairs to filter the list.
                       Looks in both attributes and dictionary values.

    Yields:
        T: Matched object or dictionary.

    Raises:
        ValueError: If no matches or multiple matches are found.
    """

    def match(config, filters):
        flat_filters = flatten_dict(filters, delimiter="/")

        for k, v in flat_filters.items():
            keys = k.split("/")
            temp = config
            for key in keys:
                if key in temp:
                    temp = temp[key]
                elif hasattr(temp, key):
                    temp = getattr(temp, key)
                else:
                    return False
            if temp != v:
                return False
        return True

    for obj in objs:
        if match(obj, filters):
            yield obj


def find_obj(objs: List[T], **filters: Any) -> T:
    """
    Find and return a single object based on filters.

    Parameters:
        objs (List[T]): List of dictionaries or objects.
        filters (Any): Nested key-value pairs to filter the list.

    Returns:
        T: Matched object or dictionary, or None if not found.
    """
    return next(filter_objs(objs, **filters))
