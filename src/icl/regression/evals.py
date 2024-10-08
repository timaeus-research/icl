import functools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from icl.regression.baselines import dmmse_predictor, ridge_predictor
from icl.regression.tasks import (DiscreteTaskDistribution,
                                  GaussianTaskDistribution,
                                  RegressionSequenceDistribution)
from infra.evals import ModelEvaluator
from infra.utils.seed import set_seed


def mse(y1, y2, axis=None):
    """
    Loss function: Mean squared error between the elements of two tensors of
    the same shape (summed along all axes or only `axis`).

    * Used as a loss function for least-squares regression
      (e.g., `mse(ys_true, ys_pred)`).
    * Used to compare the difference between two algorithms' regression
      predictions.
      (e.g., `mse(ys_algo1, ys_algo2)`).
    * If `ys1` and `ys2` are (batch, time, dimension) tensors, then we can
      get a vector of per-token losses by averaging over only the first and
      last dimensions (e.g., `mse(ys1, ys2, axis=(0, 2))`).
    """
    return (y1 - y2).square().mean(axis=axis)


class RegressionEvaluator(ModelEvaluator):
    """
    Stores fixed evaluation data batches, computed at the start of the
    training run, as well as baseline predictions for these batches.
    """
    def __init__(
        self,
        pretrain_dist: RegressionSequenceDistribution,
        true_dist: RegressionSequenceDistribution[GaussianTaskDistribution],
        max_examples: int,
        eval_batch_size: int,
        seed: int = 0
    ):
        if seed is not None:
            set_seed(seed)
 
        self.num_tasks = pretrain_dist.task_distribution.num_tasks
    
        # fixed evaluation batches (computed once at start of training run)
        self.pretrain_xs, self.pretrain_ys, self.pretrain_ws = pretrain_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
            return_ws=True,
        )
        self.true_xs, self.true_ys, self.true_ws = true_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
            return_ws=True,
        )

        # configure baseline predictors
        # ridge is the bayes-optimal predictor for the true data
        ridge = functools.partial(
            ridge_predictor,
            noise_variance=true_dist.noise_variance,
        )

        # dmmse is the bayes-optimal predictor for the pretraining data
        if self.num_tasks == math.inf:
            dmmse = ridge
        else:
            dmmse = functools.partial(
                dmmse_predictor,
                prior=pretrain_dist.task_distribution,
                noise_variance=pretrain_dist.noise_variance,
            )
     
        # cache baseline predictions (to compare against model predictions)
        self.pretrain_ridge_preds = ridge(self.pretrain_xs, self.pretrain_ys)
        self.true_ridge_preds = ridge(self.true_xs, self.true_ys)
        self.pretrain_dmmse_preds = dmmse(self.pretrain_xs, self.pretrain_ys)
        self.true_dmmse_preds = dmmse(self.true_xs, self.true_ys)
     

    @torch.no_grad()
    def __call__(self, model: nn.Module):
        """
        Evaluate a model against stored batches, returning a dictionary of
        various metrics.
        """
        # compute model predictions and loss on fixed batch from T_pretrain
        pretrain_model_preds = model(self.pretrain_xs, self.pretrain_ys)
        pretrain_model_losses = mse(self.pretrain_ys, pretrain_model_preds, axis=(0,2))
        pretrain_delta_dmmses = mse(pretrain_model_preds, self.pretrain_dmmse_preds, axis=(0,2))
        pretrain_delta_ridges = mse(pretrain_model_preds, self.pretrain_ridge_preds, axis=(0,2))
        pretrain_model_subsequence_losses = SubsequenceMSELoss()(self.pretrain_ys, pretrain_model_preds)

        # compute model predictions and loss on fixed batch from T_true
        true_model_preds = model(self.true_xs, self.true_ys)
        true_model_losses = mse(self.true_ys, true_model_preds, axis=(0,2))
        true_delta_dmmses = mse(true_model_preds, self.true_dmmse_preds, axis=(0,2))
        true_delta_ridges = mse(true_model_preds, self.true_ridge_preds, axis=(0,2))
        true_model_subsequence_losses = SubsequenceMSELoss()(self.true_ys, true_model_preds)
        # compute and return various metrics based on above

        def get_token_losses_dict(losses: torch.Tensor, label: str):
            metrics = {f"{label}/token/{i}": losses[i].item() for i in range(losses.shape[0])} 
            metrics[f"{label}"] = losses.mean().item()
            return metrics

        return {
            "pretrain/mse_subsequence": pretrain_model_subsequence_losses.mean().item(),
            **get_token_losses_dict(pretrain_model_losses, "pretrain/mse_subseq"),
            **get_token_losses_dict(pretrain_model_losses, "pretrain/mse"),
            **get_token_losses_dict(pretrain_delta_dmmses, "pretrain/delta_dmmse"),
            **get_token_losses_dict(pretrain_delta_ridges, "pretrain/delta_ridge"),
            "true/mse_subsequence": true_model_subsequence_losses.mean().item(),
            **get_token_losses_dict(true_model_losses, "true/mse"),
            **get_token_losses_dict(true_delta_dmmses, "true/delta_dmmse"),
            **get_token_losses_dict(true_delta_ridges, "true/delta_ridge"),

        }
    
    
def compose(functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

class SequenceMSELoss:
    def __init__(self, batch_reduction: str = "mean", context_reduction: str = "mean") -> None:
        self.batch_reduction = batch_reduction
        self.context_reduction = context_reduction

        mean_reduction_dims = []
        sum_reduction_dims = []

        if self.batch_reduction == 'mean':
            mean_reduction_dims.append(0)
        elif self.batch_reduction == 'sum':
            sum_reduction_dims.append(0)
        elif self.batch_reduction != 'none':
            raise ValueError(f"Unknown reduction: {self.batch_reduction}")

        if self.context_reduction == 'mean':
            mean_reduction_dims.append(1)
        elif self.context_reduction == 'sum':
            sum_reduction_dims.append(1)
        elif self.context_reduction != 'none':
            raise ValueError(f"Unknown reduction: {self.context_reduction}")      

        mean_reduction_dims.append(2)
        self.mean_reduction_dims = tuple(mean_reduction_dims)
        self.sum_reduction_dims = tuple(sum_reduction_dims)

        if self.batch_reduction == 'mean' and self.context_reduction == 'mean':
            self.loss_fn = F.mse_loss
        else:
            loss_fn = nn.MSELoss(reduction="none")
            reductions = []

            if self.batch_reduction == 'mean':
                reductions.append(lambda x: x.mean(dim=self.mean_reduction_dims))
            elif self.batch_reduction == 'sum':
                reductions.append(lambda x: x.sum(dim=self.sum_reduction_dims))
            
            if not reductions:
                self.loss_fn = loss_fn
            else:
                self.loss_fn = lambda y_pred, y: compose(reductions)(loss_fn(y_pred, y))
 

    def __call__(
            self, 
            y_pred: torch.Tensor,  # B K
            y: torch.Tensor  # B K
    ) -> torch.Tensor:
        """
        Compute the MSE loss between y_pred and y, but only on a random subsequence
        of the first K' elements of y_pred and y, where K' is sampled uniformly from
        [1, K]. 
        
        Always takes the mean over tokens. Reduction is applied to the batch.
        """
        return self.loss_fn(y_pred, y)

    
class SubsequenceMSELoss:
    def __init__(self, batch_reduction: str = "mean", context_reduction: str = "mean") -> None:
        self.batch_reduction = batch_reduction
        self.context_reduction = context_reduction

        mean_reduction_dims = []
        sum_reduction_dims = []

        if self.batch_reduction == 'mean':
            mean_reduction_dims.append(0)
        elif self.batch_reduction == 'sum':
            sum_reduction_dims.append(0)
        elif self.batch_reduction != 'none':
            raise ValueError(f"Unknown reduction: {self.batch_reduction}")

        if self.context_reduction == 'mean':
            mean_reduction_dims.append(1)
        elif self.context_reduction == 'sum':
            raise ValueError(f"Only mean reduction supported for context")
            sum_reduction_dims.append(1)
        elif self.context_reduction != 'none':
            raise ValueError(f"Unknown reduction: {self.context_reduction}")      
        else:
            raise ValueError(f"Only mean reduction supported for context")

        mean_reduction_dims.append(2)
        self.mean_reduction_dims = tuple(mean_reduction_dims)
        self.sum_reduction_dims = tuple(sum_reduction_dims)

    def __call__(
            self, 
            y_pred: torch.Tensor,  # B K
            y: torch.Tensor  # B K
    ) -> torch.Tensor:
        """
        Compute the MSE loss between y_pred and y, but only on a random subsequence
        of the first K' elements of y_pred and y, where K' is sampled uniformly from
        [1, K]. 
        
        Always takes the mean over tokens. Reduction is applied to the batch.
        """

        # Apply random mask to y_pred & y
        B, K, _ = y_pred.shape

        loss = torch.zeros(B).to(y_pred.device)

        for i in range(B):
            k = np.random.randint(1, K + 1)
            loss[i] = F.mse_loss(y_pred[i, :k], y[i, :k], reduction='mean')

        if self.batch_reduction == 'mean':
            loss = loss.mean()
        elif self.batch_reduction == 'sum':
            loss = loss.sum()

        return loss