import functools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformer_lens.utils import lm_cross_entropy_loss

from icl.constants import DEVICE, XLA
from icl.regression.baselines import dmmse_predictor, ridge_predictor
from icl.regression.tasks import (DiscreteTaskDistribution,
                                  GaussianTaskDistribution,
                                  RegressionSequenceDistribution)
from infra.evals import ModelEvaluator
from infra.utils.seed import set_seed

if XLA:
    import torch_xla.core.xla_model as xm
 
 
class LanguageEvaluator(ModelEvaluator):
    def __init__(
        self,
        testloader: torch.utils.data.DataLoader,
    ):
        self.testloader = testloader
     
    @torch.no_grad()
    def __call__(self, model: nn.Module):
        """
        Evaluate a model against stored batches, returning a dictionary of
        various metrics.
        """
        loss = torch.zeros(1, device=model.device)
        for i, batch in enumerate(self.testloader):
            if XLA: xm.mark_step()  
            x, y = batch
            x, y = x.to(model.device), y.to(model.device)
            y_pred = model(x)
            loss += lm_cross_entropy_loss(y_pred, y)
            if XLA: xm.mark_step()  

        loss /= len(self.testloader)
        return {'test/loss': loss.item()}
      