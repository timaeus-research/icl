import functools
import math

import datasets
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformer_lens.utils import lm_cross_entropy_loss

from icl.constants import DEVICE, XLA
from icl.language.data import get_loader
from infra.evals import ModelEvaluator
from infra.utils.seed import set_seed

if XLA:
    import torch_xla.core.xla_model as xm
 
 
class LanguageEvaluator(ModelEvaluator):
    def __init__(
        self,
        testset: datasets.Dataset,
        batch_size: int = 100,
    ):
        self.testset = testset
        self.batch_size = batch_size
        self.testloader = get_loader(testset, shuffle=False, batch_size=batch_size)
     
    @torch.no_grad()
    def __call__(self, model: nn.Module):
        """
        Evaluate a model against stored batches, returning a dictionary of
        various metrics.
        """
        device = next(model.parameters()).device
        loss = 0
        len_ = 0
        
        for i, batch in enumerate(self.testloader):
            tokens = batch['tokens'].to(device)
            logits = model(tokens)
            loss += lm_cross_entropy_loss(logits, tokens).item() * len(tokens)
            len_ += len(tokens)

        loss /= len_
        return {'test/loss': loss}
      