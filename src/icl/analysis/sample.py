import inspect
import itertools
import time
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm

from icl.analysis.cov import make_transformer_cov_accumulator
from icl.analysis.health import ChainHealthException
from icl.analysis.hessians import batch_hessian
from icl.analysis.sgld import SGLD
from icl.analysis.slt import (ExpectedBatchLossEstimator,
                              LikelihoodMetricsEstimator,
                              SLTObservablesEstimator)
from icl.analysis.weights import WeightsTrace
from icl.constants import DEVICE, XLA
from icl.monitoring import stdlogger
from icl.regression.evals import SequenceMSELoss, SubsequenceMSELoss
from icl.regression.train import Run
from infra.utils.iterables import dicts_to_latex

if XLA:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

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
    subsample: bool = False
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    model.train()
    pbar = tqdm(zip(range(num_steps), itertools.cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose)

    try: 
        for i, (xs, ys) in  pbar:
            optimizer.zero_grad()
            xs, ys = xs.to(device), ys.to(device)
            y_preds = model(xs, ys)
            loss = criterion(y_preds, ys)

            if subsample:
                k = np.random.randint(0, loss.numel() + 1)
                mean_loss = loss.view(-1)[:k].mean()
            else:
                mean_loss = loss.mean()

            mean_loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=mean_loss.item())

            if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
                draw = (i - num_burnin_steps) // num_steps_bw_draws

                with torch.no_grad():
                    for callback in callbacks:
                        call_with(callback, **locals())  # Cursed but we'll fix it later
                        
    except ChainHealthException as e:
        warnings.warn(f"Chain failed to converge: {e}")

def sample_single_chain_xla(
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
    device: Union[str, torch.device] = torch.device("xla"),
    callbacks: List[Callable] = [],
    subsample: bool = False
):
    xm.mark_step()
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)
    xm.mark_step()

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    model.train()

    para_loader = pl.ParallelLoader(loader, [device])
    loader = para_loader.per_device_loader(device)

    pbar = tqdm(zip(range(num_steps), itertools.cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose)
    xm.mark_step()

    try: 
        for i, (xs, ys) in  pbar:
            optimizer.zero_grad()
            xs, ys = xs.to(device), ys.to(device)
            y_preds = model(xs, ys)
            loss = criterion(y_preds, ys)

            if subsample:
                k = np.random.randint(0, loss.numel() + 1)
                mean_loss = loss.view(-1)[:k].mean()
            else:
                mean_loss = loss.mean()

            mean_loss.backward()
            optimizer.step()
            xm.mark_step()

            pbar.set_postfix(loss=mean_loss.item())

            if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
                draw = (i - num_burnin_steps) // num_steps_bw_draws

                with torch.no_grad():

                    for callback in callbacks:
                        call_with(callback, **locals())  # Cursed but we'll fix it later
            
                xm.mark_step()

    except ChainHealthException as e:
        warnings.warn(f"Chain failed to converge: {e}")


def _sample_single_chain(kwargs):
    if kwargs.get('device', torch.device('cpu')).type == "xla":
        return sample_single_chain_xla(**kwargs)
    
    return sample_single_chain(**kwargs)


def _sample_single_chain_worker(index, num_chains, get_args):
    """ Worker function for multiprocessing """
    return _sample_single_chain(get_args(index % num_chains))


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
    cores: Union[int, Literal['auto']] = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    callbacks: List[Callable] = [],
    subsample: bool =False
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
    if cores == "auto":
        cores = min(4, cpu_count())

        if XLA:
            cores = xm.xrt_world_size()
    
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
            subsample=subsample,
        )

    results = []

    if XLA:
        xmp.spawn(_sample_single_chain_worker, args=(num_chains, get_args), nprocs=cores)
    elif cores > 1:
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
    num_chains: int = 1000
    num_draws: int = 25

    # SGLD steps
    sampling_method: Literal["sgld", "sgnht"] = "sgld" # Only SGLD is supported for now
    grad_batch_origin: Literal["infinite-dataset", "eval-dataset"] = 'eval-dataset'  # Only eval-dataset is supported for now
    grad_batch_size: int = 1024

    # Parametrization 1 (original)
    epsilon: float = None
    gamma: float = None
    temperature: float = "auto"

    # Parametrization 2 (new)
    gradient_scale: float = None 
    localization_scale: float = None
    noise_scale: float = None

    # Misc
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1
    bounding_box_size: Optional[float] = None

    # SGLD evals
    eval_method: Literal["grad-minibatch", "new-minibatch", "fixed-minibatch", "dataset"] = 'grad-minibatch'
    eval_batch_size: Optional[int] = None 
    eval_dataset_size: int = 8192
    eval_metrics: List[Literal["likelihood-derived", "singular-fluctuation", "covariance", "hessian", "batch-loss", "weights"]] \
        = Field(default_factory=lambda: ["likelihood-derived"])  # covariance and hessian are not supported for now
    eval_online: bool = False
    eval_loss_fn: Literal["mse", "subsequence-mse"] = "subsequence-mse"

    num_init_loss_batches: Optional[int] = None
        
    # For Hessians and covariances
    num_evals: Optional[int] = 5

    cores: int = 1
    device: str = "cpu"
    per_token: bool = True

    init_seed: Optional[int] = None

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
        if data.get("eval_method", "grad-minibatch") in ["grad-minibatch", "new-minibatch"]:
            if "singular-fluctuation" in data.get("eval_metrics", []):
                warnings.warn("Singular fluctuation should not be trusted with minibatch evals")

            assert (
                (data.get("eval_batch_size", None) == data.get("grad_batch_size", None)) 
                or ((data.get("eval_batch_size", None) is None) != (data.get("grad_batch_size", None) is None))
            ), "Eval batch size must match grad batch size for minibatch evals"

            # assert not bool(data.get("eval_batch_size", None)) and not bool(data.get("grad_batch_size", None)), "Eval batch size or grad batch size is required for minibatch evals"

            data["eval_batch_size"] = data.get("eval_batch_size", None) or data.get("grad_batch_size", 1024)
            data["grad_batch_size"] = data.get("grad_batch_size", None) or data.get("eval_batch_size", 1024)

        elif data["eval_method"] == "fixed-minibatch":
            assert data.get("eval_batch_size", None) is not None, "Eval batch size is required for minibatch evals"
        else:
            if data.get("eval_batch_size", None) is not None: 
                warnings.warn("Eval batch size is provided but will be ignored for dataset evals")

        assert "covariance" not in data.get("eval_metrics", []), "Covariance is not supported for now"

        # Parametrization
        num_samples = data.get("eval_dataset_size", 2**20)

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
            data["temperature"] = temperature = (gradient_scale * 2 / (epsilon * num_samples)) ** -1
            data["gamma"] = gamma = localization_scale * 2 / epsilon

        else:
            if temperature == "auto":
                data["temperature"] = temperature = 1 / np.log(num_samples)
            
            data["gradient_scale"] = gradient_scale = epsilon * temperature * num_samples / 2
            data["localization_scale"] = localization_scale = epsilon * gamma / 2
            data["noise_scale"] = noise_scale = epsilon
   
        return data

    def to_sampler(self, run: Run, log_fn: Optional[Callable] = None):
        return Sampler(self, run, log_fn=log_fn)
        
    def get_loss_fn(self, variant="mse", batch_reduction: str = "mean", context_reduction: str = "mean"):
        if variant == "mse":
            return SequenceMSELoss(batch_reduction=batch_reduction, context_reduction=context_reduction)
        else:
            return SubsequenceMSELoss(batch_reduction=batch_reduction, context_reduction="mean")

    def get_optimizer_cls(self):
        if self.sampling_method == "sgld":
            return SGLD
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
    
    def to_latex(self):
        return dicts_to_latex({
            r"\epsilon": str(self.epsilon),
            r"\beta": str(self.temperature ** -1),
            r"\gamma": str(self.gamma),
            "eval": (self.eval_method, self.eval_loss_fn),
        })

class Sampler:
    def __init__(self, config: SamplerConfig, run: Run, log_fn: Optional[Callable] = None):
        self.config = config
        self.run = run

        xs, ys = run.pretrain_dist.get_batch(
            num_examples=run.config.task_config.max_examples,
            batch_size=self.config.eval_dataset_size,
        )

        xs.to(DEVICE)
        ys.to(DEVICE)

        self.full_dataset = torch.utils.data.TensorDataset(xs, ys)
        self.eval_dataset = self.full_dataset

        if self.config.eval_method == "fixed-minibatch":
            self.eval_dataset = torch.utils.data.TensorDataset(xs[:self.config.eval_batch_size, :, :], ys[:self.config.eval_batch_size, :, :])

        self.grad_loader = torch.utils.data.DataLoader(self.full_dataset, batch_size=self.config.grad_batch_size, shuffle=True)  # Shuffle might meant repeats
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.config.eval_batch_size, shuffle=(self.config.eval_method == "new-minibatch"))

        self.log_fn = log_fn
        context_reduction = "none" if self.config.per_token else "mean"
        self.grad_loss_fn = self.config.get_loss_fn("mse", batch_reduction="mean", context_reduction=context_reduction)  # mse-subsequence is passed to the sample function
        self.eval_loss_fn = self.config.get_loss_fn(
            self.config.eval_loss_fn if not self.config.per_token else 'mse', 
            batch_reduction="none" if "singular-fluctuation" in self.config.eval_metrics else "mean", 
            context_reduction=context_reduction
        )
        if XLA:
            xm.mark_step()

        self.init_loss = self.eval_model(run.model, max_num_batches=self.config.num_init_loss_batches, verbose=True)
        
        if XLA:
            xm.mark_step()
            
        self._callbacks = self.get_callbacks()
        self.update_init_loss(self.init_loss)

    def eval_one_batch(self, model):
        xs, ys = next(iter(self.eval_loader))
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        y_preds = model(xs, ys)
        return self.eval_loss_fn(y_preds, ys).detach()

    def iter_eval_model(self, model): 
        for xs, ys in self.eval_loader:
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            y_preds = model(xs, ys)
            yield self.eval_loss_fn(y_preds, ys).detach()

    def eval_model(self, model, max_num_batches: Optional[int] = None, verbose=False):
        loss = None

        for i, (xs, ys) in tqdm(enumerate(self.eval_loader), desc="Evaluating init loss", total = (max_num_batches or len(self.eval_loader)), disable=not verbose):
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            y_preds = model(xs, ys)
            _loss = self.eval_loss_fn(y_preds, ys)

            loss = loss if loss is not None else torch.zeros_like(_loss)
            loss += _loss.detach() * xs.shape[0]

            if max_num_batches and max_num_batches <= i:
                break

        if loss is None:
            raise ValueError("No batches in eval loader")

        return (loss / self.config.eval_dataset_size).detach()
        
    def get_cov_callback(self):
        return make_transformer_cov_accumulator(self.run.model, device=DEVICE, num_evals=self.config.num_evals)

    def get_likelihood_metrics_callback(self):
        loss_fn = None  # self.config.eval_method == "grad-minibatch"

        if self.config.eval_method == "new-minibatch":
            loss_fn = self.eval_one_batch
        elif self.config.eval_method in ("fixed-minibatch", "dataset"):
            loss_fn = self.eval_model

        return LikelihoodMetricsEstimator(
            self.config.num_chains, 
            self.config.num_draws,
            dataset_size=self.config.eval_dataset_size,
            temperature=self.config.temperature,
            loss_fn=loss_fn,
            device=DEVICE,
            online=self.config.eval_online,
            include_trace=self.config.eval_online,
            log_fn=self.log_fn,
            init_loss=self.init_loss
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
            device=DEVICE,
            online=self.config.eval_online,
            include_trace=self.config.eval_online,
            log_fn=self.log_fn,
            init_loss=self.init_loss
        )
    
    def get_batch_loss_callback(self):
        return ExpectedBatchLossEstimator(
            self.config.num_chains, 
            self.config.num_draws, 
            self.loss_dim,
            DEVICE,
            online=True,
            include_trace=True
        )

    def get_weights_callback(self):
        return WeightsTrace(
            self.config.num_chains, 
            self.config.num_draws, 
            self.run.model, 
            DEVICE,
        )

    def get_callbacks(self):
        callbacks = {}

        if "likelihood-derived" in self.config.eval_metrics and "singular-fluctuation" in self.config.eval_metrics:
            callbacks['slt'] = self.get_slt_callback()
        elif "likelihood-derived" in self.config.eval_metrics:
            callbacks['likelihood'] = self.get_likelihood_metrics_callback()
        elif "singular-fluctuation" in self.config.eval_metrics:
            raise ValueError("Singular fluctuation requires likelihood-derived")
        if "batch-loss" in self.config.eval_metrics:
            callbacks['batch-loss'] = self.get_batch_loss_callback()
        if "weights" in self.config.eval_metrics:
            callbacks['weights'] = self.get_weights_callback()
    
        return callbacks
    
    def eval_hessians(self, model: nn.Module):
        stdlogger.info("Evaluating hessians...")
        start = time.perf_counter()
        model.zero_grad()

        xs, ys = self.full_dataset.tensors[0][:self.config.eval_batch_size], self.full_dataset.tensors[1][:self.config.eval_batch_size]

        with batch_hessian(model, xs, ys) as H:
            results = {
                "hessian/trace": H.trace(),
                "hessian/eigenvals": H.eigenvalues(top_n=self.config.num_evals)[0]
            }

        end = time.perf_counter()
        stdlogger.info(f"Evaluated hessians in {end - start:.2f}s")
        return results

    def eval(self, model: nn.Module, seed=None):
        results = sample(
            model,
            self.grad_loader,
            self.grad_loss_fn,
            self.config.get_optimizer_cls(),
            optimizer_kwargs=self.config.get_optimizer_kwargs(),
            num_draws=self.config.num_draws,
            num_chains=self.config.num_chains,
            cores=self.config.cores,
            device=DEVICE,
            callbacks=self.callbacks,
            seed=seed,
            subsample=self.config.eval_loss_fn == "subsequence-mse"
        )

        if "hessian" in self.config.eval_metrics:
            results.update(self.eval_hessians(model))

        return results
    
    def reset(self):
        for callback in self.callbacks:
            callback.reset()

    def update_init_loss(self, init_loss):
        stdlogger.info("Updating init loss to %s", init_loss)
        self.init_loss = init_loss

        for callback in self.callbacks:
            if hasattr(callback, "init_loss"):
                callback.init_loss = init_loss

        if self.config.init_seed is not None:
            torch.manual_seed(self.config.init_seed)

    @property
    def callbacks(self):
        return list(self._callbacks.values())
    
    @property
    def batch_loss(self):
        if 'batch-loss' not in self._callbacks:
            raise ValueError("Batch loss not enabled in config")

        return self._callbacks['batch-loss']
    
    @property
    def weights(self):
        if 'weights' not in self._callbacks:
            raise ValueError("Weights not enabled in config")
        
        return self._callbacks['weights']
    
    @property
    def slt(self):
        if 'slt' not in self._callbacks:
            raise ValueError("SLT not enabled in config")
        
        return self._callbacks['slt']
    
    @property
    def likelihood(self):
        if 'likelihood' not in self._callbacks:
            raise ValueError("Likelihood not enabled in config")
        
        return self._callbacks['likelihood']
    
    @property
    def loss_dim(self):
        return self.init_loss.numel()