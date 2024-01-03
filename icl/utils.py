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


def get_locations(L: int):
    locations = [
        'token_sequence_transformer.token_embedding',
    ]

    for l in range(L):
        locations.extend([
            f'token_sequence_transformer.blocks.{l}',
            f'token_sequence_transformer.blocks.{l}.attention.attention',
            f'token_sequence_transformer.blocks.{l}.compute',
        ])

    locations.extend([
        'token_sequence_transformer.unembedding',
    ])

    return locations


def get_model_locations_to_display(L: int):
    locations = {
        'token_sequence_transformer.token_embedding': "Embedding",
    }

    for l in range(L):
        locations.update({
            f'token_sequence_transformer.blocks.{l}': f"Block {l}",
            f'token_sequence_transformer.blocks.{l}.attention.attention': f"Block {l} Attention Logits",
            f'token_sequence_transformer.blocks.{l}.compute': f"Block {l} MLP",
        })

    locations.update({
        'token_sequence_transformer.unembedding': "Unembedding",
    })

    return locations


def get_model_locations_to_slug(L: int):
    locations = {
        'token_sequence_transformer.token_embedding': "0.0-embed",
    }

    for l in range(L):
        locations.update({
            f'token_sequence_transformer.blocks.{l}': f"{l+1}-block",
            f'token_sequence_transformer.blocks.{l}.attention.attention': f"{l+1}.1-attn",
            f'token_sequence_transformer.blocks.{l}.compute': f"{l+1}.2-mlp",
        })

    locations.update({
        'token_sequence_transformer.unembedding': f"{L+1}-unembed-ln",
    })

    return locations


def prepare_experiments():
    from icl.constants import ANALYSIS, FIGURES

    load_dotenv()
    sns.set_theme(style="whitegrid")

    assert os.path.exists(FIGURES)
    assert os.path.exists(ANALYSIS)

    logging.basicConfig(level=logging.INFO)
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
    else:
        return obj


def get_device(obj):
    """
    Recursively get the device of tensors in a nested object.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, (dict, OrderedDict)):
        return next(d for d in (get_device(v) for v in obj.values()) if d is not None)
    elif isinstance(obj, (list, tuple)):
        return next(d for d in (get_device(v) for v in obj) if d is not None)
    else:
        return None


@contextlib.contextmanager
def to_device(state_dicts, device='cpu'):
    original_device = get_device(state_dicts)

    if str(device) != str(original_device):
        logging.info("Moving state dicts to %s.", device)

    try:
        move_to_device(state_dicts, device)
        # Yield control back to the with block
        yield state_dicts
    finally:
        if str(device) != str(original_device):
            logging.info("Moving state dicts back to %s.", original_device)

        move_to_device(state_dicts, original_device)
