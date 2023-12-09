import inspect
import itertools
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import numpy as np
import torch
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from icl.analysis.cov import make_transformer_cov_accumulator
from icl.analysis.slt import (LikelihoodMetricsEstimator,
                              SLTObservablesEstimator)
from icl.evals import SequenceMSELoss, SubsequenceMSELoss
from icl.train import Run


def call_with(func: Callable, **kwargs):
    """Check the func annotation and call with only the necessary kwargs."""
    sig = inspect.signature(func)
    
    # Filter out the kwargs that are not in the function's signature
    if "kwargs" in sig.parameters:
        filtered_kwargs = kwargs

    else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Call the function with the filtered kwargs
    return func(**filtered_kwargs)


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    verbose=True,
    device: Union[str, torch.device] = torch.device("cpu"),
    callbacks: List[Callable] = [],
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    model.train()

    for i, (xs, ys) in  tqdm(zip(range(num_steps), itertools.cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose):
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs, ys)
        loss = criterion(y_preds, ys)

        loss.backward()
        optimizer.step()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw = (i - num_burnin_steps) // num_steps_bw_draws
            loss = loss.item()

            with torch.no_grad():
                for callback in callbacks:
                    call_with(callback, **locals())  # Cursed but we'll fix it later


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    callbacks: List[Callable] = [],
):
    """
    Sample model weights using a given optimizer, supporting multiple chains.

    Parameters:
        model (torch.nn.Module): The neural network model.
        step (Literal['sgld']): The name of the optimizer to use to step.
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        num_draws (int): Number of samples to draw.
        num_chains (int): Number of chains to run.
        num_burnin_steps (int): Number of burn-in steps before sampling.
        num_steps_bw_draws (int): Number of steps between each draw.
        cores (Optional[int]): Number of cores for parallel execution.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling.
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the optimizer.
    """
    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            verbose=verbose,
            callbacks=callbacks,
        )

    results = []

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            results = pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            results.append(_sample_single_chain(get_args(i)))
    
    results = {}

    for callback in callbacks:
        if hasattr(callback, "estimate"):
            results.update(callback.estimate())

    return results


class SamplerConfig(BaseModel):
    # Sampling
    num_chains: int
    num_draws: int

    # SGLD steps
    sampling_method: Literal["sgld", "sgnht"]  # Only SGLD is supported for now
    grad_batch_origin: Literal["infinite-dataset", "eval-dataset"]  # Only eval-dataset is supported for now
    grad_batch_size: int

    # Parametrization 1 (original)
    epsilon: float 
    gamma: float 
    temperature: float = "auto"

    # Parametrization 2 (new)
    gradient_scale: float
    localization_scale: float
    noise_scale: float

    # Misc
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1
    bounding_box_size: Optional[float] = None

    # SGLD evals
    eval_method: Literal["grad-minibatch", "new-minibatch", "fixed-minibatch", "dataset"]
    eval_batch_size: Optional[int] = None
    eval_dataset_size: int = 8192
    eval_metrics: List[Literal["likelihood-derived", "singular-fluctuation", "covariance", "hessian"]] \
        = Field(default_factory=lambda: ["likelihood-derived", "singular-fluctuation"])  # covariance and hessian are not supported for now
    eval_online: bool = False
    eval_loss_fn: Literal["mse", "subsequence-mse"] = "subsequence-mse"
        
    # Covariance estimation
    num_evals: Optional[int] = None

    cores: int = 1
    device: str = "cpu"

    @field_validator('sampling_method')
    @classmethod
    def check_sampling_method(cls, v: str) -> str:
        assert v == "sgld", "Only SGLD is supported for now"
        return v

    @field_validator('grad_batch_origin')
    @classmethod
    def check_grad_batch_origin(cls, v: str) -> str:
        assert v  == "eval-dataset", "Only eval-dataset is supported for now"
        return v

    # Validate all fields
    @model_validator(mode='before')
    @classmethod
    def check_evals(cls, data: Any) -> Any:
        if data["eval_method"] in ["grad-minibatch", "new-minibatch"]:
            if "singular-fluctuation" in data["eval_metrics"]:
                warnings.warn("Singular fluctuation should not be trusted with minibatch evals")

            assert data.get("eval_batch_size", None) is not None, "Eval batch size is required for minibatch evals"
        elif data["eval_method"] == "fixed-minibatch":
            assert data.get("eval_batch_size", None) is not None, "Eval batch size is required for minibatch evals"
        else:
            if data.get("eval_batch_size", None) is not None: 
                warnings.warn("Eval batch size is provided but will be ignored for dataset evals")

        assert "covariance" not in data["eval_metrics"], "Covariance is not supported for now"
        assert "hessian" not in data["eval_metrics"], "Hessian is not supported for now"

        # Parametrization
        num_samples = data["eval_dataset_size"]

        temperature = data.get("temperature", None)
        gamma = data.get("gamma", None)
        epsilon = data.get("epsilon", None)

        gradient_scale = data.get("gradient_scale", None)
        localization_scale = data.get("localization_scale", None)
        noise_scale = data.get("noise_scale", None)

        assert ((epsilon is None) and (temperature is None or temperature == "auto") and (gamma is None)) or \
            ((noise_scale is None) and (gradient_scale is None) and (localization_scale is None)), f"Must choose and stick to one parametrization, received: epsilon={epsilon}, temperature={temperature}, gamma={gamma}, gradient_scale={gradient_scale}, localization_scale={localization_scale}, noise_scale={noise_scale}"

        if epsilon is None:
            data["epsilon"] = epsilon = noise_scale
            data["temperature"] = temperature = gradient_scale * 2 / (epsilon * num_samples)
            data["gamma"] = gamma = localization_scale * 2 / epsilon

        else:
            if temperature == "auto":
                data["temperature"] = temperature = 1 / np.log(data["eval_dataset_size"])
            
            data["gradient_scale"] = gradient_scale = epsilon * temperature * num_samples / 2
            data["localization_scale"] = localization_scale = epsilon * gamma / 2
            data["noise_scale"] = noise_scale = epsilon
   
        return data

    def to_sampler(self, run: Run, log_fn: Optional[Callable] = None):
        return Sampler(self, run, log_fn=log_fn)
        
    def get_loss_fn(self, reduction: str = "mean"):
        if self.eval_loss_fn == "mse":
            return SequenceMSELoss(reduction=reduction)
        else:
            return SubsequenceMSELoss(reduction=reduction)

    def get_optimizer_cls(self):
        if self.sampling_method == "sgld":
            return SGLD
        elif self.sampling_method == "sgnht":
            return SGNHT
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def get_optimizer_kwargs(self):
        return {
            "lr": self.epsilon,
            "elasticity": self.gamma,
            "temperature": self.temperature,
            "bounding_box_size": self.bounding_box_size,
            "num_samples": self.eval_dataset_size,
        }

class Sampler:
    def __init__(self, config: SamplerConfig, run: Run, log_fn: Optional[Callable] = None):
        self.config = config
        self.run = run

        xs, ys = run.pretrain_dist.get_batch(
            num_examples=run.config.task_config.max_examples,
            batch_size=self.config.eval_dataset_size,
        )

        self.full_dataset = torch.utils.data.TensorDataset(xs, ys)
        self.eval_dataset = self.full_dataset

        if self.config.eval_method == "fixed-minibatch":
            self.eval_dataset = torch.utils.data.Subset(self.full_dataset, range(self.config.eval_batch_size))

        self.grad_loader = torch.utils.data.DataLoader(self.full_dataset, batch_size=self.config.grad_batch_size, shuffle=True)  # Shuffle might meant repeats
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.config.eval_batch_size, shuffle=(self.config.eval_method == "new-minibatch"))

        self.log_fn = log_fn
        self.grad_loss_fn = self.config.get_loss_fn(reduction="mean")
        self.eval_loss_fn = self.config.get_loss_fn(reduction="none" if "singular-fluctuation" in self.config.eval_metrics else "mean")
        self.callbacks = self.get_callbacks()

    def eval_one_batch(self, model):
        xs, ys = next(iter(self.eval_loader))
        xs, ys = xs.to(self.config.device), ys.to(self.config.device)
        y_preds = model(xs, ys)
        return self.eval_loss_fn(y_preds, ys)

    def iter_eval_model(self, model):
        for xs, ys in self.eval_loader:
            xs, ys = xs.to(self.config.device), ys.to(self.config.device)
            y_preds = model(xs, ys)
            yield self.eval_loss_fn(y_preds, ys)

    def eval_model(self, model):
        return sum(self.iter_eval_model(model)) / self.config.eval_dataset_size
        
    def get_cov_callback(self):
        return make_transformer_cov_accumulator(self.run.model, device=self.config.device, num_evals=self.config.num_evals)

    def get_likelihood_metrics_callback(self):
        loss_fn = None  # self.config.eval_method == "grad-minibatch"

        if self.config.eval_method == "new-minibatch":
            loss_fn = self.eval_one_batch
        elif self.config.eval_method in ("fixed-minibatch", "dataset"):
            loss_fn = self.eval_model

        if self.log_fn is not None:
            warnings.warn("Log function is not supported for likelihood metrics")

        return LikelihoodMetricsEstimator(
            self.config.num_chains, 
            self.config.num_draws,
            dataset_size=self.config.eval_dataset_size,
            temperature=self.config.temperature,
            loss_fn=loss_fn,
            device=self.config.device,
            online=self.config.eval_online,
            include_trace=self.config.eval_online,
        )

    def get_slt_callback(self):
        if self.config.eval_method not in ("fixed-minibatch", "dataset"):
            warnings.warn("Singular fluctuation should not be trusted with minibatch evals")

        return SLTObservablesEstimator(
            self.config.num_chains, 
            self.config.num_draws,
            self.config.eval_dataset_size,
            self.iter_eval_model,
            temperature=self.config.temperature,
            device=self.config.device,
            online=self.config.eval_online,
            include_trace=self.config.eval_online,
            log_fn=self.log_fn,
        )

    def get_callbacks(self):
        callbacks = []

        if "likelihood-derived" in self.config.eval_metrics and "singular-fluctuation" in self.config.eval_metrics:
            callbacks.append(self.get_slt_callback())
        elif "likelihood-derived" in self.config.eval_metrics:
            callbacks.append(self.get_likelihood_metrics_callback())
        elif "singular-fluctuation" in self.config.eval_metrics:
            raise ValueError("Singular fluctuation requires likelihood-derived")
    
        return callbacks

    def eval(self, model: nn.Module):
        return sample(
            model,
            self.grad_loader,
            self.grad_loss_fn,
            self.config.get_optimizer_cls(),
            optimizer_kwargs=self.config.get_optimizer_kwargs(),
            num_draws=self.config.num_draws,
            num_chains=self.config.num_chains,
            cores=self.config.cores,
            device=self.config.device,
            callbacks=self.callbacks,
        )
    
    def reset(self):
        for callback in self.callbacks:
            callback.reset()