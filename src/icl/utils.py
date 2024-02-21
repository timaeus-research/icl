import contextlib
import logging
import os
from collections import OrderedDict

import pandas as pd
import seaborn as sns
import sentry_sdk
import torch
from dotenv import load_dotenv

from icl.monitoring import stdlogger


def prepare_experiments():
    from icl.constants import ANALYSIS, FIGURES

    load_dotenv()
    sns.set_theme(style="ticks")

    assert os.path.exists(FIGURES)
    assert os.path.exists(ANALYSIS)

    # set_start_method('spawn')  # Required for sharing CUDA tensors
    sentry_sdk.init(
        dsn="https://92ea29f1e366cda4681fb10273e6c2a7@o4505805155074048.ingest.sentry.io/4505805162479616",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )


def get_default_device(device=None):
    """
    Returns the default device for PyTorch.
    """

    device = os.environ.get("DEVICE", device)

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm
        stdlogger.info("Using TPU.")
        return xm.xla_device()
    except (ModuleNotFoundError, ImportError):
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")

    stdlogger.warning("No GPU found, falling back to CPU.")
    return torch.device("cpu")



def move_to_device(obj, device):
    """
    Recursively move tensors in a nested object to the specified device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, OrderedDict):  
        return OrderedDict((k, move_to_device(v, device)) for k, v in obj.items())
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        raise ValueError(f"Unknown type: {type(obj)}")


def get_device(obj):
    """
    Get the first device of tensors in a nested object via DFS.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, (dict, OrderedDict)):
        return next(d for d in (get_device(v) for v in obj.values()) if d is not None)
    elif isinstance(obj, (list, tuple)):
        return next(d for d in (get_device(v) for v in obj) if d is not None)
    else:
        return None

