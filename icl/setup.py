import os
from logging import getLogger

import torch

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
    except ModuleNotFoundError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    
    stdlogger.warning("No GPU found, falling back to CPU.")
    return torch.device("cpu")


DEVICE = get_default_device()
XLA = DEVICE.type == "xla"