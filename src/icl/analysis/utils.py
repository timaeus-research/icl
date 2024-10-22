
import functools
import re
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from icl.regression.config import get_config
from icl.regression.train import RegressionRun
from infra.evals import ModelEvaluator
from infra.integrations.wandb import generate_config_dicts_from_path
from infra.io.storage import BaseStorageProvider
from infra.utils.iterables import (filter_objs, find_obj, find_unique_obj,
                                   flatten_dict)


def get_run(sweep: str, **filters):
    """
    Find the run with the specified filters in the specified sweep.
    Returns the first run that matches the filters.
    """
    config_dicts = list(generate_config_dicts_from_path(sweep))
    config_dict = find_obj(config_dicts, **filters) 
    config = get_config(**config_dict)
    run = RegressionRun.create_and_restore(config)
    return run


def get_unique_config(sweep: str, **filters):
    """
    Find the config with the specified filters in the specified sweep.
    Requires that only one config matches the filters.
    """
    config_dicts = list(generate_config_dicts_from_path(sweep))
    config_dict = find_unique_obj(config_dicts, **filters) 
    config = get_config(**config_dict)
    return config


def get_unique_run(sweep: str, **filters):
    """
    Find the run with the specified filters in the specified sweep.
    Requires that only one run matches the filters.
    """
    config = get_unique_config(sweep, **filters)
    run = RegressionRun.create_and_restore(config)
    return run


def get_sweep_configs(sweep: str, **filters):
    """
    Generate RegressionConfigs for all runs in the specified sweep.
    """
    for sweep_config_dict in filter_objs(generate_config_dicts_from_path(sweep), **filters):
        yield get_config(**sweep_config_dict)


def wandb_run_to_df(run):
    history_df = run.history()
    config_dict = get_config(**run.config).model_dump()

    for k, v in run.config.items():
        if k not in config_dict:
            config_dict[k] = v

    del config_dict["logger_config"]
    del config_dict["checkpointer_config"]

    config_dict_flat = flatten_dict(config_dict, flatten_lists=True)
    
    for k, v in config_dict_flat.items():
        if isinstance(v, tuple):
            # Repeat the tuple for the entire length of the DataFrame
            v = [v] * len(history_df)
            
        history_df[k] = v

    return history_df


def wandb_runs_to_df(runs):
    return pd.concat([wandb_run_to_df(run) for run in tqdm(runs, desc="Converting runs to dfs")])


def load_model_at_step(config, step: int, checkpointer: Optional[BaseStorageProvider] = None):
    if checkpointer is None:
        checkpointer = config.checkpointer_config.factory()

    model = config.task_config.model_factory()
    model_state_dict = checkpointer.load_file(step)["model"]
    model.load_state_dict(model_state_dict)

    return model


def load_model_at_last_checkpoint(config, checkpointer: Optional[BaseStorageProvider] = None):
    if checkpointer is None:
        checkpointer = config.checkpointer_config.factory()

    model = config.task_config.model_factory()
    model_state_dict = checkpointer[-1]["model"]
    model.load_state_dict(model_state_dict)

    return model
    

def map_evals_over_checkpoints(model, checkpointer: BaseStorageProvider, evaluator: ModelEvaluator, verbose=False):
    steps = checkpointer.file_ids

    for step in tqdm(steps, disable=not verbose):
        model_state_dict = checkpointer.load_file(step)["model"]
        model.load_state_dict(model_state_dict)
        yield {**evaluator(model), "step": step}


def split_attn_weights(W: torch.Tensor, num_heads: int, embed_dim: int, head_size: int):
    W_split = W.view((embed_dim, num_heads, head_size * 3))

    for h in range(num_heads):
        yield tuple(W_split[:, h, i*head_size:(i+1)*head_size] for i in range(3))


def get_weights(model, paths):
    for path in paths:
        full_path = path.split(".")
        layer = model

        for p in full_path:
            layer = getattr(layer, p)

        yield layer.weight.view((-1,))

        if layer.bias is not None:
            yield layer.bias.view((-1,))


def log_on_update(callback, monitor, log_fn):
    @functools.wraps(callback.update)
    def update(*args, **kwargs):
        callback.update(*args, **kwargs)
        result = monitor(callback, *args, **kwargs)

        if result:
            log_fn(result)
    
        return result
    
    callback.update = update
    return callback


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
    pattern = escaped_template.split('**')
    pattern = [p.replace('*', '[^.]*') for p in pattern]
    pattern = '.*'.join(pattern)
    pattern = f'^{pattern}$'
    return re.match(pattern, string) is not None
    
    

