import logging
from typing import Iterable, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel, model_validator

OptimizerTypeLiteral = Literal["SGD", "Adam", "AdamW", "SGLD"]


def optimizer_factory(parameters: Iterable[torch.nn.Parameter], optimizer_type: OptimizerTypeLiteral, **kwargs):
    if optimizer_type == "SGD":
        return torch.optim.SGD(parameters, **kwargs)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(parameters, **kwargs)
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(parameters, **kwargs)
    elif optimizer_type == "SGLD":
        from icl.analysis.sgld import SGLD
        
        return SGLD(parameters, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    

class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerTypeLiteral = "SGD"
    lr: float = 0.01
    weight_decay: float = 0.0001
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    noise_level: Optional[float] = None
    elasticity: Optional[float] = None
    temperature: Optional[Union[Literal["adaptive"], float]] = None
    num_samples: Optional[int] = None  # If -1, then this needs to be filled in later.

    class Config:
        frozen = True

    def model_dump(self, *args, **kwargs):
        # Only export relevant fields based on optimizer_type
        fields = {"optimizer_type", "lr", "weight_decay"}
        if self.optimizer_type == "SGD":
            fields.add("momentum")
        elif self.optimizer_type in {"Adam", "AdamW"}:
            fields.add("betas")
        elif self.optimizer_type == "SGLD":
            fields.update({"noise_level", "elasticity", "temperature", "num_samples"})

        return super().model_dump(include=fields, *args, **kwargs)

    @model_validator(mode="after")
    def validate_optimizer_type(self) -> "OptimizerConfig":
        if self.optimizer_type == "SGLD":
            assert (
                self.noise_level is not None
            ), "noise_level must be specified for SGLD"
            assert self.elasticity is not None, "elasticity must be specified for SGLD"
            assert (
                self.temperature is not None
            ), "temperature must be specified for SGLD"
            assert (
                self.num_samples is not None
            ), "num_samples must be specified for SGLD"
        elif self.optimizer_type == "SGD":
            assert self.momentum is not None, "momentum must be specified for SGD"
        elif self.optimizer_type in {"Adam", "AdamW"}:
            assert self.betas is not None, "betas must be specified for Adam/AdamW"

        return self

    def factory(self, parameters: Iterable[torch.nn.Parameter]):
        optimizer_type = self.optimizer_type
        optimizer_params = self.model_dump(exclude={"optimizer_type"})
        return optimizer_factory(parameters, optimizer_type, **optimizer_params)
