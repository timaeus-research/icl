from typing import Any, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from devinfra.utils.device import get_default_device
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from pydantic import BaseModel, model_validator, field_validator, Field
from torch.nn import functional as F

import wandb
from icl.analysis.cov import make_transformer_cov_accumulator
from icl.analysis.sample import estimate_slt_observables
from icl.config import ICLConfig, get_config
from icl.evals import ICLEvaluator
from icl.experiments.utils import *
from icl.train import Run
from icl.utils import pyvar_dict_to_latex, pyvar_dict_to_slug

app = typer.Typer()

class SamplerConfig(BaseModel):
    # Sampling
    num_chains: int
    num_draws: int

    # SGLD steps
    sampling_method: Literal["sgld", "sgnht"]
    grad_batch_origin: Literal["infinite-dataset", "eval-dataset"]
    grad_batch_size: int

    # Sets the absolute scale of the SGLD steps
    epsilon: float 

    # Parametrization 1 (original)
    gamma: Optional[float] = None
    temperature: Optional[float] = None

    # Parametrization 2 (new)
    gradient_scale: Optional[float] = None
    localization_scale: Optional[float] = None
    noise_scale: float = 1.

    bounding_box_size: Optional[float] = None

    # SGLD evals
    eval_method: Literal["grad-minibatch", "new-minibatch", "dataset", "validation-set"]
    eval_batch_size: Optional[int] = None
    eval_dataset_size: int = 8192
    eval_metrics: List[Literal["likelihood-derived", "singular-fluctuation", "covariance", "hessian"]] = Field(default_factory=lambda: ["likelihood-derived", "singular-fluctuation"])
    eval_online: bool = False

    # Covariance estimation
    num_evals: int = 3

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
            assert "singular-fluctuation" not in data["eval_metrics"], "Singular fluctuation is not supported for minibatch evals"
            assert data.get("eval_batch_size", None) is not None, "Eval batch size is required for minibatch evals"
        else:
            assert data.get("eval_batch_size", None) is None, "Eval batch size is not supported for dataset/validation-set evals"

        assert "covariance" not in data["eval_metrics"], "Covariance is not supported for now"
        assert "hessian" not in data["eval_metrics"], "Hessian is not supported for now"

        # Gradient term
        assert (data["gradient_scale"] is None) != (data["temperature"] is None), "Exactly one of gradient_scale and temperature must be specified"
        # TODO: convert
        if data["temperature"] is None:
            # data["temperature"] = 1. / np.log(data["eval_dataset_size"])
            data["temperature"] = None
        else:
            data["gradient_scale"] = None        

        # Localization term
        assert (data["localization_scale"] is None) != (data["gamma"] is None), "Exactly one of localization_scale and gamma must be specified"

        # TODO
        if data["gamma"] is None:
            data["gamma"] = None 
        else: 
            data["localization_scale"] = None


def sweep_over_final_weights(
    config: dict,
    sampler_config: dict,
):      
    cores = int(os.environ.get("CORES", 1))
    device = get_default_device()

    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)

    sampler_config: SamplerConfig = SamplerConfig(**sampler_config)
    
    # Evals
    xs, ys = run.pretrain_dist.get_batch(
        num_examples=run.config.task_config.max_examples,
        batch_size=sampler_config.eval_dataset_size,
    )
    dataset = torch.utils.data.TensorDataset(xs, ys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=sampler_config.eval_batch_size)  # Shuffle might meant repeats

    # Hyperparameters for posterior sampling
    num_chains = sampler_config.pop("num_chains", 25)
    num_draws = sampler_config.pop("num_draws", 1000)
    sampling_method = sampler_config.pop("sampling_method", "sgld")

    if sampling_method == "sgld":
        optimizer_class = SGLD
    elif sampling_method == "sgnht":
        optimizer_class = SGNHT
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # Covariance estimation
    callbacks = []
    num_evals = sampler_config.pop("num_evals", 0)
    if num_evals > 0:
        cov_accumulator = make_transformer_cov_accumulator(run.model, device=device, num_evals=num_evals)
        callbacks.append(cov_accumulator)

    # Sample observables
    results = estimate_slt_observables(
        run.model,
        loader,
        F.mse_loss,
        optimizer_class,
        optimizer_kwargs=dict(
            **sampler_config,
            batch_size=batch_size,
            num_samples=eff_num_samples,
        ),
        num_draws=num_draws,
        num_chains=num_chains,
        cores=cores,
        device=device,
        callbacks=callbacks,
        online="observed"
    )

    # Save to wandb
    wandb.log(results)

    # Save locally
    results["config"] = config
    results["sampler_config"] = sampler_config
    slug = "llc-" + pyvar_dict_to_slug({
        "num_layers": config.task_config.num_layers,
        "num_heads": config.task_config.num_heads,
        "num_tasks": config.task_config.num_tasks,
        "num_draws": num_draws,
        **sampler_config,
    }) + ".pt"

    torch.save(results, ANALYSIS / slug)

       

@app.command("wandb")
def wandb_sweep_over_final_weights():      
    wandb.init(project="icl-llc", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("analysis_config")
    title_config = sampler_config.copy()
    del title_config["num_draws"]
    del title_config["num_chains"]
    del title_config["batch_size"]
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}:{pyvar_dict_to_slug(title_config)}"
    wandb.run.save()
    sweep_over_final_weights(config, sampler_config)
    wandb.finish()


if __name__ == "__main__":
    prepare_experiments()
    app()
    


