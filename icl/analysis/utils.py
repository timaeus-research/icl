
from devinfra.integrations.wandb import generate_config_dicts_from_path
from devinfra.utils.iterables import find_obj, find_unique_obj

from icl.config import get_config
from icl.train import Run


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