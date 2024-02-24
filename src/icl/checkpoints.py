from typing import Dict, List, TypedDict

import torch


class StateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Dict
    rng_state: List
    metadata: Dict


def state_dict(model, optimizer, scheduler, rng_state: torch.Tensor, **metadata) -> StateDict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {k: v for k,v in scheduler.state_dict().items() if not callable(v)} if scheduler is not None else None,  # Required because of the custom scheduler
        "rng_state": rng_state.tolist(),  # Required because of the custom scheduler,
        "metadata": metadata
    }