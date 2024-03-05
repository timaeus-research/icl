import os
from dataclasses import asdict, is_dataclass
from typing import Any, List, Optional

import datasets
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from transformer_lens import HookedTransformer, HookedTransformerConfig

import wandb
from icl.language.data import get_tokenized_dataset
from icl.monitoring import stdlogger
from infra.evals import CriterionLiteral
from infra.io import CheckpointerConfig, MetricLoggingConfig
from infra.monitoring import expand_steps_config_
from infra.optim import OptimizerConfig, SchedulerConfig
from infra.utils.iterables import (dict_to_slug, dicts_to_latex, hash_dict,
                                   nested_update)


def get_model_cfg(
        n_layers=2, 
        d_model=256, 
        d_head=32, 
        n_heads=8, 
        n_ctx=1024, 
        d_vocab=5000, 
        tokenizer_name='georgeyw/TinyStories-tokenizer-5k', 
        normalization_type='LN', 
        attn_only=True, 
        seed=0, 
        positional_embedding_type='shortformer'
):
    return HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        tokenizer_name=tokenizer_name,
        normalization_type=normalization_type,
        attn_only=attn_only,
        seed=seed,
        positional_embedding_type=positional_embedding_type,
    )


class LanguageConfig(BaseModel):
    run_name: str 
    num_training_samples: int = None
    num_steps: int = 50_000
    batch_size: int = 100
    transformer_config: HookedTransformerConfig = Field(default_factory=get_model_cfg)
    optimizer_config: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(
        optimizer_type="AdamW", lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05,
    ))
    logger_config: Optional[MetricLoggingConfig] = None
    checkpointer_config: Optional[CheckpointerConfig] = None
    scheduler_config: Optional[SchedulerConfig] = None
    trainset: str = 'timaeus/dsir-pile-10m'
    testset: str = 'georgeyw/dsir-pile-10k'

    class Config:
        arbitrary_types_allowed=True

    @field_validator("transformer_config", mode="before", check_fields=False)
    @classmethod
    def validate_transformer_config(cls, v):
        if is_dataclass(v):
            return v
        elif isinstance(v, dict):
            return get_model_cfg(**v)
        raise ValueError("Invalid configuration for HookedTransformerConfig")

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
        from pprint import pp
        pp(data)
        # Num samples
        data["num_training_samples"] = data.get("num_training_samples", data["batch_size"] * data["num_steps"])
        data["num_steps"] = data.get("num_steps", data["num_training_samples"] // data["batch_size"])

        # Automatically fill in the project_dir field of the checkpointer
        checkpoint_config = data.get("checkpointer_config", None)
        if checkpoint_config is not None:
            run_name = data['run_name']
            seed = data["transformer_config"]["seed"]

            greeks = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
            if "{" in run_name:
                run_name = run_name.format(seed_greek=greeks[seed])

            checkpoint_config["project_dir"] = data['run_name'] = run_name

        return data

    def transformer_factory(self):
        return HookedTransformer(self.transformer_config)

    def to_latex(self):
        return dicts_to_latex({
            'L': self.transformer_config.n_layers, 
            'H': self.transformer_config.n_heads, 
        }, {
            'K': self.transformer_config.n_ctx,
            'D': self.transformer_config.d_vocab,
            r'd_{\mathrm{mlp}}': self.transformer_config.d_mlp,
            r'd_{\mathrm{embed}}': self.transformer_config.d_model,
            r'\mathrm{seed}': (self.transformer_config.seed),
        }, {
            r'\eta': self.optimizer_config.lr,
            'B': self.batch_size,
            'T': self.num_steps,
        })
    
    def to_slug(self, delimiter="-", equal_sign=""):
        return dict_to_slug({
            'L': self.transformer_config.n_layers, 
            'H': self.transformer_config.n_heads, 
            'K': self.transformer_config.n_ctx,
            'D': self.transformer_config.d_vocab,
            'dmlp': self.transformer_config.d_mlp,
            'dembed': self.transformer_config.d_model,
            'seed': self.transformer_config.seed,
            'lr': self.optimizer_config.lr,
            'B': self.batch_size,
            'T': self.num_steps,
        }, delimiter=delimiter, equal_sign=equal_sign)
    
    @property
    def is_wandb_enabled(self):
        return self.logger_config is not None and self.logger_config.project is not None and self.logger_config.entity is not None

    def trainset_factory(self):
        return get_tokenized_dataset(self.trainset, self.transformer_config.tokenizer_name, self.transformer_config.n_ctx, streaming=False)
    
    def testset_factory(self):  
        return get_tokenized_dataset(self.testset, self.transformer_config.tokenizer_name, self.transformer_config.n_ctx, streaming=False)
    
    def model_dump(self):
        return {
            "run_name": self.run_name,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "transformer_config": {k: v for k, v in asdict(self.transformer_config).items() if k in ["n_layers", "n_heads", "n_ctx", "d_vocab", "d_model", "seed"]},
            "optimizer_config": self.optimizer_config.model_dump(),
            # "logger_config": self.logger_config.model_dump() if self.logger_config is not None else None,
            # "checkpointer_config": self.checkpointer_config.model_dump() if self.checkpointer_config is not None else None,
            # "scheduler_config": self.scheduler_config.model_dump() if self.scheduler_config is not None else None,
        }

def get_config(
    project: Optional[str] = None, entity: Optional[str] = None, **kwargs
) -> LanguageConfig:
    stdlogger.info("Configuring training run...")

    config_dict = {
        # evaluation config
        "run_name": "tetrahedron-3m-{seed_greek}",
        "num_steps": 50_000,
        "batch_size": 100,
        "transformer_config": {
            "n_layers": 2,
            "seed": 0
        },
        "checkpointer_config": {
            "checkpoint_steps": {"log_space": 100, "linear_space": 2_000},
            "bucket_name": os.environ['AWS_LANGUAGE_BUCKET_NAME']
        },
        # for wandb?
        "logger_config": {
            "logging_steps": {
                "log_space": 100,
                "linear_space": 100,
            },
            "project": project,
            "entity": entity,
        },
    }

    nested_update(config_dict, kwargs)        
    logger_config = config_dict["logger_config"]

    # Sync with wandb (side-effects!)
    if logger_config["project"] is not None and logger_config["entity"] is not None:
        stdlogger.info("Pulling configuration from wandb...")

        if "run_name" in config_dict:
            run_name = config_dict["run_name"]
            wandb.init(
                project=logger_config["project"],
                entity=logger_config["entity"],
                name=run_name,
            )
        else:
            wandb.init(
                project=logger_config["project"], entity=logger_config["entity"]
            )

        nested_update(config_dict, wandb.config)
        
    stdlogger.info("Preparing configuration object")
    config = LanguageConfig(**config_dict)

    if config.is_wandb_enabled:
        stdlogger.info("Updating wandb run name and config...")
        config_dict = config.model_dump()
        wandb.config.update(config_dict)
        wandb.run.name = config_dict["run_name"]

    return config