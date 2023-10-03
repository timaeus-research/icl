import functools

import torch
from devinfra.evals import ModelEvaluator
from devinfra.utils.seed import set_seed
from torch import nn

from icl.baselines import dmmse_predictor, ridge_predictor
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)


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


class ICLEvaluator(ModelEvaluator):
    """
    Stores fixed evaluation data batches, computed at the start of the
    training run, as well as baseline predictions for these batches.
    """
    def __init__(
        self,
        pretrain_dist: RegressionSequenceDistribution[DiscreteTaskDistribution],
        true_dist: RegressionSequenceDistribution[GaussianTaskDistribution],
        max_examples: int,
        eval_batch_size: int,
        seed: int = 0
    ):
        if seed is not None:
            set_seed(seed)
 
        # fixed evaluation batches (computed once at start of training run)
        self.pretrain_xs, self.pretrain_ys = pretrain_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
        )
        self.true_xs, self.true_ys = true_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
        )

        # configure baseline predictors
        # dmmse is the bayes-optimal predictor for the pretraining data
        dmmse = functools.partial(
            dmmse_predictor,
            prior=pretrain_dist.task_distribution,
            noise_variance=pretrain_dist.noise_variance,
        )

        # ridge is the bayes-optimal predictor for the true data
        ridge = functools.partial(
            ridge_predictor,
            noise_variance=true_dist.noise_variance,
        )

        # cache baseline predictions (to compare against model predictions)
        self.pretrain_dmmse_preds = dmmse(self.pretrain_xs, self.pretrain_ys)
        self.pretrain_ridge_preds = ridge(self.pretrain_xs, self.pretrain_ys)
        self.true_dmmse_preds = dmmse(self.true_xs, self.true_ys)
        self.true_ridge_preds = ridge(self.true_xs, self.true_ys)


    @torch.no_grad()
    def __call__(self, model: nn.Module):
        """
        Evaluate a model against stored batches, returning a dictionary of
        various metrics.
        """
        # compute model predictions and loss on fixed batch from T_pretrain
        pretrain_model_preds = model(self.pretrain_xs, self.pretrain_ys)
        pretrain_model_losses = mse(self.pretrain_ys, pretrain_model_preds, axis=(0,2))
        # compute model predictions and loss on fixed batch from T_true
        true_model_preds = model(self.true_xs, self.true_ys)
        true_model_losses = mse(self.true_ys, true_model_preds, axis=(0,2))
        # compute and return various metrics based on above

        def get_token_losses_dict(losses: torch.Tensor, label: str):
            return {f"{label}/token/{i}": losses[i].item() for i in range(losses.shape[0])}

        return {
            "pretrain/mse": pretrain_model_losses.mean().item(),
            "pretrain/delta_dmmse": mse(pretrain_model_preds, self.pretrain_dmmse_preds),
            "pretrain/delta_ridge": mse(pretrain_model_preds, self.pretrain_ridge_preds),
            **get_token_losses_dict(pretrain_model_losses, "pretrain"),
            "true/mse": true_model_losses.mean().item(),
            "true/delta_dmmse": mse(true_model_preds, self.true_dmmse_preds),
            "true/delta_ridge": mse(true_model_preds, self.true_ridge_preds),
            **get_token_losses_dict(true_model_losses, "true"),

        }