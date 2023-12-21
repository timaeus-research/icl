import logging
import os
from logging import getLogger

import seaborn as sns
import sentry_sdk
import torch
from dotenv import load_dotenv

from icl.constants import ANALYSIS, FIGURES

stdlogger = getLogger('icl')


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
        import torch_xla
        import torch_xla.core.xla_model as xm
        stdlogger.info("Using TPU.")
        return xm.xla_device()
    except (ModuleNotFoundError, ImportError):
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    
    stdlogger.warning("No GPU found, falling back to CPU.")
    return torch.device("cpu")


DEVICE = get_default_device()
XLA = DEVICE.type == "xla"


def prepare_experiments():
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