import itertools

import yaml
from devinterp.utils import flatten_dict

from icl.config import get_config
from icl.train import Run
from icl.utils import find_obj, find_unique_obj, unflatten_dict


def generate_config_dicts_from_path(file_path: str, **kwargs):
    """Load the ICLConfigs for each of the runs defined in a wandb sweep config at the specified file path."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    yield from generate_config_dicts(config, **kwargs)


def expand_grid(params):
    """Generates a list of dicts for each run config based on the grid of parameter values."""
    keys, value_sets = zip(*params)
    for values in itertools.product(*value_sets):
        yield dict(zip(keys, values))


def _wandb_config_expansion(parameters, prefix="", sep="/"):
    """Recursive function to expand nested parameters."""
    keys = list(parameters.keys())
    for key in keys:
        if "parameters" in parameters[key]:
            yield from _wandb_config_expansion(
                parameters[key]["parameters"], prefix=f"{prefix}{key}{sep}"
            )
        else:
            if "values" in parameters[key]:
                yield (f"{prefix}{key}", parameters[key]["values"])
            else:
                yield (f"{prefix}{key}", [parameters[key]["value"]])


def generate_config_dicts(sweep_config: dict, **kwargs):
    """Turns a wandb sweep config into a list of configs for each run defined in that sweep. (Assumes strategy is grid)"""
    params = list(_wandb_config_expansion(sweep_config["parameters"]))
    kwargs = flatten_dict(kwargs, delimiter="/")

    for config_dict in expand_grid(params):
        _kwargs = kwargs.copy()
        _kwargs.update(config_dict)

        yield unflatten_dict(_kwargs, delimiter="/")


def get_run(sweep: str, **filters):
    """
    Find the run with the specified filters in the specified sweep.
    Returns the first run that matches the filters.
    """
    config_dicts = list(generate_config_dicts_from_path(sweep))
    config_dict = find_obj(config_dicts, **filters) 
    config = get_config(**config_dict)
    run = Run.create_and_restore(config)
    return run


def get_unique_run(sweep: str, **filters):
    """
    Find the run with the specified filters in the specified sweep.
    Requires that only one run matches the filters.
    """
    config_dicts = list(generate_config_dicts_from_path(sweep))
    config_dict = find_unique_obj(config_dicts, **filters) 
    config = get_config(**config_dict)
    run = Run.create_and_restore(config)
    return run