import inspect
import itertools
import re
import time
import warnings
from copy import deepcopy
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple, Type,
                    Union)

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader, Subset
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
from icl.regression.train import RegressionRun
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


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def sample_single_chain(
    model: nn.Module,
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
    subsample: bool = False,
    cores=1,
):
    # Initialize new model and optimizer for this chain
    model = model.to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method((p for p in model.parameters() if p.requires_grad), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    model.train()
    pbar = tqdm(zip(range(num_steps), cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose)

    print("Optimizing over:")
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            for n, p in model.named_parameters():
                if p is param:
                    print("\t", n, f"({p.shape})")

    try: 
        if verbose:
            print(f"Starting chain {chain} on {device} with {cores} cores and {num_steps} steps.")
            start = time.time()
            
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

        if verbose:
            end = time.time()
            print(f"Chain {chain} on {device} with {cores} cores finished in {end - start:.2f}s")                    
    
    except ChainHealthException as e:
        warnings.warn(f"Chain failed to converge: {e}")


def sample_single_chain_xla(
    model: nn.Module,
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
    subsample: bool = False,
    cores=1,
):
    # Initialize new model and optimizer for this chain
    model = model.train().to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    chain_loss = torch.zeros(1, device=device)
    chain_loss_sq = torch.zeros(1, device=device)

    def increment_loss(loss):
        nonlocal chain_loss
        nonlocal chain_loss_sq
        chain_loss += loss
        chain_loss_sq += loss ** 2

    try: 
        start = time.time()
        
        for i, (xs, ys) in enumerate(cycle(loader)):   
            if i >= num_steps:
                break

            xs, ys = xs.to(device), ys.to(device)
            y_preds = model(xs, ys)
            loss = criterion(y_preds, ys)

            if subsample:
                k = np.random.randint(0, loss.numel() + 1)
                mean_loss = loss.view(-1)[:k].mean()
            else:
                mean_loss = loss.mean()

            optimizer.zero_grad()


            mean_loss.backward()
            optimizer.step()
            xm.mark_step()

            if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:

                with torch.no_grad():
                    xm.add_step_closure(increment_loss, (mean_loss, ))
            
        end = time.time()
        duration = end - start

    except ChainHealthException as e:
        warnings.warn(f"Chain failed to converge: {e}")

    xm.mark_step()

    chain_loss_mean = (chain_loss / num_draws).item()
    chain_loss_std = ((chain_loss_sq / num_draws - chain_loss_mean ** 2) ** 0.5).item()

    return {
        "loss/mean": chain_loss_mean,
        "loss/std": chain_loss_std,
        "duration": duration
    }


def _sample_single_chain(kwargs):
    if kwargs.get('device', torch.device('cpu')).type == "xla":
        kwargs['device'] = xm.xla_device() 
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
            model=deepcopy(model.to('cpu')),
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
            cores=cores
        )

    if cores > 1: 
        raise NotImplementedError("Multiprocessing is not currently supported")
        # if XLA:
        #     xmp.spawn(_sample_single_chain_worker, args=(num_chains, get_args), nprocs=cores)
        # else:
        #     ctx = get_context("spawn")
        #     with ctx.Pool(cores) as pool:
        #         pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        results = []

        for i in range(num_chains):
            results.append(_sample_single_chain(get_args(i)))
    
    if results and any(results):
        keys = list(results[0].keys())

        dataset_size = callbacks[0].dataset_size
        temperature = callbacks[0].temperature
        init_loss  = callbacks[0].init_loss.item()

        for result in results:
            result['wbic/mean'] = dataset_size * result['loss/mean']
            result['wbic/std'] = dataset_size * result['loss/std']
            result['llc/mean'] = (result['wbic/mean'] - init_loss * dataset_size) / temperature
            result['llc/std'] = result['wbic/std'] / temperature

        keys = list(results[0].keys())
        d = {
            f"{key}/{i}": result[key] for key in keys for i, result in enumerate(results)
        }
        
        for key in keys:
            entries = [result[key] for result in results]
            d[f"{key}/mean"] = float(np.mean(entries))
            d[f"{key}/std"] = float(np.std(entries))

        return d

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
    grad_batch_origin: Literal["infinite-dataset", "eval-dataset"] = 'infinite-dataset'  # Only eval-dataset is supported for now
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
    per_token: bool = False

    init_seed: Optional[int] = None

    # Restricted LLC
    include: Tuple[str, ...] = ("*", )
    exclude: Tuple[str, ...] = ()

    @field_validator('sampling_method')
    @classmethod
    def check_sampling_method(cls, v: str) -> str:
        assert v == "sgld", "Only SGLD is supported for now"
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

    def to_sampler(self, run: RegressionRun, log_fn: Optional[Callable] = None):
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
    

def match_template(template, string):
    """
    Check if a string matches a template with wildcards.

    The template string can contain two types of wildcards:
    - Single wildcard (*): Matches any character except a dot (.).
    - Double wildcard (**): Matches any character including dots.

    Parameters:
        template (str): The template string with wildcards.
        string (str): The string to match against the template.

    Returns:
        bool: True if the string matches the template, False otherwise.

    Example:
        template = 'token_sequence_transformer.blocks.*.attention.**'
        string1 = 'token_sequence_transformer.blocks.0.attention.attention.weight'
        string2 = 'token_sequence_transformer.blocks.1.ffn.weight'
        string3 = 'token_sequence_transformer.blocks.2.attention.bias'

        print(match_template(template, string1))  # Output: True
        print(match_template(template, string2))  # Output: False
        print(match_template(template, string3))  # Output: True
    """

    escaped_template = template.replace('.', '\\.')
    pattern = escaped_template.replace('*', '[^.]*')
    pattern = pattern.replace('**', '.*')
    pattern = f'^{pattern}$'
    return re.match(pattern, string) is not None


class Sampler:
    def __init__(self, config: SamplerConfig, run: RegressionRun, log_fn: Optional[Callable] = None, device: Optional[torch.device] = None):
        self.config = config
        self.run = run
        self.device = device or DEVICE

        if config.init_seed is not None:
            torch.manual_seed(config.init_seed)

        # if XLA:
        #     self.device = torch.device('cpu')  # Excuse me
        
        if self.config.grad_batch_origin == "infinite-dataset":
            self.full_dataset, self.grad_loader = self.run.pretrain_dist.as_dataset_and_loader(
                self.run.config.task_config.max_examples,
                self.config.grad_batch_size
            ) 
            self.eval_dataset, self.eval_loader = self.run.pretrain_dist.as_dataset_and_loader(
                self.run.config.task_config.max_examples,
                self.config.eval_batch_size or self.config.grad_batch_size,
                self.config.eval_dataset_size,
            )
        else:
            self.full_dataset, self.grad_loader = self.run.pretrain_dist.as_dataset_and_loader(
                self.run.config.task_config.max_examples,
                self.config.grad_batch_size,
                self.config.eval_dataset_size
            )
            self.eval_dataset = self.full_dataset
            self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.config.eval_batch_size, shuffle=(self.config.eval_method == "new-minibatch"))

        if self.config.eval_method == "fixed-minibatch":
            warnings.warn("Fixed minibatch evals are not supported for now.")

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

            if XLA:
                xm.mark_step()

    def eval_model(self, model, max_num_batches: Optional[int] = None, verbose=False):
        loss = None

        with torch.no_grad():
            for i, (xs, ys) in tqdm(enumerate(self.eval_loader), desc="Evaluating init loss", total = (max_num_batches or len(self.eval_loader)), disable=not verbose):
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                y_preds = model(xs, ys)
                _loss = self.eval_loss_fn(y_preds, ys)

                loss = loss if loss is not None else torch.zeros_like(_loss, device=DEVICE)
                loss += _loss.detach() * xs.shape[0]

                if max_num_batches and max_num_batches <= i:
                    break

            if loss is None:
                raise ValueError("No batches in eval loader")

            return (loss / self.config.eval_dataset_size).detach()
        
    def get_cov_callback(self):
        return make_transformer_cov_accumulator(self.run.model, device=self.device, num_evals=self.config.num_evals)

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
            device=self.device,
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
            self.device,
            online=True,
            include_trace=True
        )

    def get_weights_callback(self):
        return WeightsTrace(
            self.config.num_chains, 
            self.config.num_draws, 
            self.run.model, 
            self.device,
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

    def restrict_(self, model: nn.Module):
        """
        Restricts the gradients of the model's parameters based on the include and exclude configurations.

        Exclusive patterns take precedence over inclusive patterns if a parameter matches both. 

        Args:
            model (nn.Module): The model whose parameters' gradients need to be restricted.

        Returns:
            None
        """
        include_templates = [e for e in self.config.include if "*" in e]
        include_exact = [e for e in self.config.include if "*" not in e]

        exclude_templates = [e for e in self.config.exclude if "*" in e]
        exclude_exact = [e for e in self.config.exclude if "*" not in e]

        for name, param in model.named_parameters():
            param.requires_grad = False

            if any(name == e for e in include_exact):
                param.requires_grad = True

            elif any(match_template(e, name) for e in include_templates):
                param.requires_grad = True               

            if any(name == e for e in exclude_exact):
                param.requires_grad = False
            elif any(match_template(e, name) for e in exclude_templates):
                param.requires_grad = False

    def eval(self, model: nn.Module, seed=None):
        if "*" not in self.config.include or self.config.exclude:
            self.restrict_(model)

        results = sample(
            model,
            self.grad_loader,
            self.grad_loss_fn,
            self.config.get_optimizer_cls(),
            optimizer_kwargs=self.config.get_optimizer_kwargs(),
            num_draws=self.config.num_draws,
            num_chains=self.config.num_chains,
            num_burnin_steps=self.config.num_burnin_steps,
            num_steps_bw_draws=self.config.num_steps_bw_draws,
            cores=self.config.cores,
            device=self.device,
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