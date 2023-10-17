import math
from typing import Any, Optional

from devinfra.evals import CriterionLiteral
from devinfra.io import CheckpointerConfig, MetricLoggingConfig
from devinfra.monitoring import expand_steps_config_
from devinfra.optim import OptimizerConfig, SchedulerConfig
from devinfra.utils.device import get_default_device
from devinfra.utils.iterables import (dict_to_slug, dicts_to_latex, hash_dict,
                                      nested_update)
from devinfra.utils.seed import set_seed
from pydantic import BaseModel, Field, model_validator
import torch # This is for the type-hint for xm_device

import wandb
from icl.model import InContextRegressionTransformer
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)


class ICLTaskConfig(BaseModel):
    # paper notation in comments

    task_size: int = 8 # D, dimensions of linear regression task
    max_examples: int = 16 # K, in-context examples (thus max_context = 2*K)
    num_tasks: int     # M, task-diversity of pre-train dist
    noise_variance: float = 0.25 # sigma^2 i.e. y = wx + N(0, sigma^2)
    embed_size: int = 128 # d_e = d_mid (in Phuong notation)
    mlp_size: int = 128 # two layer ReLU network with 128 nodes in hidden layer (layer sizes [d_e, mlp_size, d_e])
    num_heads: int = 2 # attention heads per layer 
    num_layers: int = 8 # each layer has one attention head and one MLP 
    use_mlp: bool = True
    use_layernorm: bool = True
    use_softmax: bool = True
    task_init: str = 'normal' # method for instantiating tasks, 'basis' or 'normal'
    method_params: dict = {} # parameters for task distribution initialisation
    model_seed: int = 0 # random seed 
    pretrain_seed: int = 1 
    true_seed: int = 2
    sampling_seed: int = 3

    def model_factory(self):
        if self.model_seed is not None:
            set_seed(self.model_seed)

        # Anything that is changed about the model must then be changed in dtransformer.py and model.py
        return InContextRegressionTransformer(
            task_size=self.task_size,
            max_examples=self.max_examples,
            embed_size=self.embed_size,
            mlp_size=self.mlp_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            use_mlp =self.use_mlp,
            use_layernorm=self.use_layernorm,
            use_softmax=self.use_softmax,
        )

    def pretrain_dist_factory(self):
        if self.pretrain_seed is not None:
            set_seed(self.pretrain_seed)

        return RegressionSequenceDistribution(
            task_distribution=DiscreteTaskDistribution(
                num_tasks=self.num_tasks,
                task_size=self.task_size,
                task_init=self.task_init,
                method_params=self.method_params
            ),
            noise_variance=self.noise_variance,
        )

    def true_dist_factory(self):
        # No need to set the seed here (that comes at when sampling)
        return RegressionSequenceDistribution(
            task_distribution=GaussianTaskDistribution(
                task_size=self.task_size,
            ),
            noise_variance=self.noise_variance,
        )


class ICLConfig(BaseModel):
    # Dataset & loader
    num_training_samples: int
    batch_size: int = 128
    run_name: Optional[str] = None

    # Training loop
    # num_epochs: int = None
    num_steps: int = 100_000
    logger_config: Optional[MetricLoggingConfig] = None
    checkpointer_config: Optional[CheckpointerConfig] = None

    # Optimizer
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler_config: Optional[SchedulerConfig] = None

    # Misc
    device: str = Field(default_factory=get_default_device)
    criterion: CriterionLiteral = "cross_entropy"
    xm_device: "torch.device" = torch.device("cpu") # THIS COULD BE BAD

    eval_batch_size: int
    task_config: ICLTaskConfig

    class Config:
        # frozen = True # So that I can update the xm_device later on
        arbitrary_types_allowed = True

    # Properties

    @property
    def num_steps_per_epoch(self):
        """Number of steps per epoch."""
        return self.num_training_samples // self.batch_size

    @property
    def num_epochs(self):
        """Number of epochs."""
        return math.ceil(self.num_steps / self.num_steps_per_epoch)

    @property
    def is_wandb_enabled(self):
        """Whether wandb is enabled."""
        return self.logger_config and self.logger_config.is_wandb_enabled

    # Validators

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, data: Any):
        num_steps = data["num_steps"]

        checkpoint_config = data.get("checkpointer_config", None)
        logger_config = data.get("logger_config", None)

        # Automatically expand `checkpoint_steps` for checkpointer and `logging_steps` for logger
        # "log_space": 10 -> "log_space": [1, num_steps, 10]
        checkpoint_steps = checkpoint_config.get("checkpoint_steps", None)
        if isinstance(checkpoint_steps, dict):
            expand_steps_config_(checkpoint_steps, num_steps)

        # Logger
        logger_steps = logger_config.get("logging_steps", None)
        if isinstance(logger_steps, dict):
            expand_steps_config_(logger_steps, num_steps)

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_extra(cls, data: Any):
        num_tasks = data["task_config"]["num_tasks"]
        num_steps = data["num_steps"]

        # Num samples
        data["num_training_samples"] = num_steps * data["batch_size"]

        # LR Scheduler
        optimizer_config = data.get("optimizer_config", None)
        scheduler_config = data.get("scheduler_config", None)
        if scheduler_config is not None:
            scheduler_config["max_lr"] = scheduler_config.get(
                "max_lr", optimizer_config.get("lr", 1e-3)
            )
            scheduler_config["total_steps"] = scheduler_config.get(
                "max_steps", num_steps
            )
            scheduler_config["div_factor"] = scheduler_config.get(
                "div_factor", (num_steps / 2 - 1)
            )
            scheduler_config["final_div_factor"] = scheduler_config.get(
                "final_div_factor", (num_steps / 2 - 1)
            )

        # Automatically fill in the project_dir field of the checkpointer
        checkpoint_config = data.get("checkpointer_config", None)
        if num_tasks is not None and checkpoint_config is not None:
            task_config_dict = data["task_config"]
            task_config_hash = hash_dict(task_config_dict)[:6]
            opt_config_hash = hash_dict(data["optimizer_config"])[:6]
            scheduler_config_hash = hash_dict(data["scheduler_config"])[:6]
            run_name = f"ntasks-{num_tasks}-task-{task_config_hash}-opt-{opt_config_hash}-sched-{scheduler_config_hash}"
            checkpoint_config["project_dir"] = checkpoint_config.get(
                "project_dir", f"icl/{run_name}"
            )
            data["run_name"] = run_name

            if "extra" in data:
                data["run_name"] += f"-{data.pop('extra')}"

        return data

    def to_latex(self):
        return dicts_to_latex({
            'L': self.task_config.num_layers, 
            'H': self.task_config.num_heads, 
            'M': self.task_config.num_tasks,
        }, {
            'K': self.task_config.max_examples,
            'D': self.task_config.task_size,
            r'\sigma^2': self.task_config.noise_variance,
            r'd_{\mathrm{mlp}}': self.task_config.mlp_size,
            r'd_{\mathrm{embed}}': self.task_config.embed_size,
            r'\mathrm{seeds}': (self.task_config.model_seed, self.task_config.pretrain_seed, self.task_config.true_seed, self.task_config.sampling_seed),
        }, {
            'n': self.num_training_samples,
            r'\eta': self.optimizer_config.lr,
            'B': self.batch_size,
            'T': self.num_training_samples // self.batch_size,
        })
    
    def to_slug(self, delimiter="-", equal_sign=""):
        return dict_to_slug({
            'L': self.task_config.num_layers, 
            'H': self.task_config.num_heads, 
            'M': self.task_config.num_tasks,
            'K': self.task_config.max_examples,
            'D': self.task_config.task_size,
            'err': self.task_config.noise_variance,
            'dmlp': self.task_config.mlp_size,
            'dembed': self.task_config.embed_size,
            'seeds': delimiter.join(map(str, [self.task_config.model_seed, self.task_config.pretrain_seed, self.task_config.true_seed, self.task_config.sampling_seed])),
            'n': self.num_training_samples,
            'lr': self.optimizer_config.lr,
            'B': self.batch_size,
            'T': self.num_training_samples // self.batch_size,
        }, delimiter=delimiter, equal_sign=equal_sign)

def get_config(
    project: Optional[str] = None, entity: Optional[str] = None, **kwargs
) -> ICLConfig:
    # (shared parameters)
    num_steps = 2**18
    batch_size = 256
    max_learning_rate = 1e-3
    num_tasks = 64

    config_dict = {
        # model & data config
        "task_config": {
            "num_tasks": num_tasks,
        },
        # training config
        "num_steps": num_steps,
        "batch_size": batch_size,
        "optimizer_config": {
            "optimizer_type": "Adam",
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "lr": max_learning_rate,  # unused (overwritten by scheduler)
        },
        "scheduler_config": {
            "scheduler_type": "OneCycleLR",
            "anneal_strategy": "linear",
            "pct_start": 0.5,
            "cycle_momentum": False,  # N/A but required to avoid error
        },
        # evaluation config
        "eval_batch_size": 2048,
        "checkpointer_config": {
            "checkpoint_steps": {"log_space": 50, "linear_space": 50},
            "bucket_name": "devinterp",
            # "local_root": "./checkpoints",
        },
        # for wandb?
        "logger_config": {
            "logging_steps": {
                "log_space": 1000,
                "linear_space": 1000,
            },
            "project": project,
            "entity": entity,
            # "stdout": True
        },
        "device": 'xla',
    }

    nested_update(config_dict, kwargs)        
    logger_config = config_dict["logger_config"]

    # Sync with wandb (side-effects!)
    if logger_config["project"] is not None and logger_config["entity"] is not None:
        if "run_name" in config_dict:
            run_name = config_dict.pop("run_name")
            wandb.init(
                project=logger_config["project"],
                entity=logger_config["entity"],
                name=run_name,
            )
        else:
            wandb.init(
                project=logger_config["project"], entity=logger_config["entity"]
            )

        # d2 overrides d1 in nested_update, so all the wandb settings from the .yaml ultimately override the defaults
        nested_update(config_dict, wandb.config)
        
    return ICLConfig(**config_dict)
