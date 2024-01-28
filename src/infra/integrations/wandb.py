import itertools

import yaml

from infra.utils.iterables import flatten_dict, unflatten_dict


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

