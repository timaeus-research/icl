import random
from typing import Protocol

import numpy as np
import torch
import torch.backends

from infra.utils.device import DeviceOrDeviceLiteral


class Seedable(Protocol):
    def set_seed(self, seed: int):
        ...


def set_seed(seed: int, *seedables: Seedable):
    """
    Sets the seed for the Learner.

    Args:
        seed (int): Seed to set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    for seedable in seedables:
        seedable.set_seed(seed)

    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except ImportError:
        pass

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
