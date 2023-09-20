import hashlib
import json
import random
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Union

import numpy as np
import torch


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