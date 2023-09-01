from typing import Any, Optional

from devinterp.learner import LearnerConfig
from pydantic import BaseModel, model_validator

from icl.model import InContextRegressionTransformer
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)
from icl.utils import hash_dict


class ICLTaskConfig(BaseModel):
    task_size: int
    max_examples: int
    num_tasks: int
    noise_variance: float
    embed_size: int
    mlp_size: int
    num_heads: int
    num_layers: int

    def model_factory(self):
        return InContextRegressionTransformer(
            task_size=self.task_size,
            max_examples=self.max_examples,
            embed_size=self.embed_size,
            mlp_size=self.mlp_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        )

    def pretrain_dist_factory(self):
        return RegressionSequenceDistribution(
            task_distribution=DiscreteTaskDistribution(
                num_tasks=self.num_tasks,
                task_size=self.task_size,
            ),
            noise_variance=self.noise_variance,
        )

    def true_dist_factory(self):
        return RegressionSequenceDistribution(
            task_distribution=GaussianTaskDistribution(
                task_size=self.task_size,
            ),
            noise_variance=self.noise_variance,
        )


class ICLConfig(LearnerConfig):
    eval_batch_size: int
    task_config: ICLTaskConfig

    @model_validator(mode='before')
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
            scheduler_config["max_lr"] = scheduler_config.get("max_lr", optimizer_config.get("lr", 1e-3))
            scheduler_config["total_steps"] = scheduler_config.get("max_steps", num_steps)
            scheduler_config["div_factor"] = scheduler_config.get("div_factor", (num_steps/2 - 1))
            scheduler_config["final_div_factor"] = scheduler_config.get("final_div_factor", (num_steps/2 - 1))

        # Automatically fill in the project_dir field of the checkpointer
        checkpoint_config = data.get("checkpointer_config", None)
        if num_tasks is not None and checkpoint_config is not None:
            task_config_hash = hash_dict(data["task_config"])[:6]
            opt_config_hash = hash_dict(data["optimizer_config"])[:6]
            scheduler_config_hash = hash_dict(data["scheduler_config"])[:6]
            checkpoint_config["project_dir"] = checkpoint_config.get("project_dir", f"icl/ntasks-{num_tasks}-task-{task_config_hash}-opt-{opt_config_hash}-sched-{scheduler_config_hash}")

        return data


def get_config(project: Optional[str] = None, entity: Optional[str] = None) -> ICLConfig:
    # (shared parameters)
    num_steps = 500_000
    batch_size = 256
    max_learning_rate = 1e-3
    num_tasks = 64

    config_dict = {
        # model & data config
        "task_config": {
            "task_size": 8,
            "max_examples": 16,
            "num_tasks": num_tasks,
            "noise_variance": 0.25,
            "embed_size": 128,
            "mlp_size": 128,
            "num_heads": 2,
            "num_layers": 8,
        },
        # training config
        "num_steps": num_steps,
        "batch_size": batch_size,
        "optimizer_config": {
            "optimizer_type": "Adam",
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "lr": max_learning_rate,   # unused (overwritten by scheduler)
        },
        "scheduler_config": {
            "scheduler_type": "OneCycleLR",
            "anneal_strategy": 'linear',
            "pct_start": 0.5,
            "cycle_momentum": False,    # N/A but required to avoid error
        },
        # evaluation config
        "eval_batch_size": 2048,
        "checkpointer_config": {
            "checkpoint_steps": {
                "log_space": 50,
                "linear_space": 50
            },
            "bucket_name": "devinterp",
            # "local_root": "./checkpoints",
        },
        # for wandb?
        "logger_config": {
            "logging_steps": {
                "log_space": 100,
                "linear_space": 100,
            },
            "project": project,
            "entity": entity,
            # "stdout": True
        }
    }

    return ICLConfig(**config_dict)