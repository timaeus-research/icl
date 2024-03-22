from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Literal, Protocol, Set, Tuple,
                    Union)

from infra.utils.iterables import int_linspace, int_logspace

StepsConfig = Dict[Literal["log_space", "linear_space"], Tuple[int, int, int]]
StepsConfigShortened = Dict[Literal["log_space", "linear_space"], int]

class Monitor(Protocol):
    """Defines the protocol for a Monitor object. A Monitor object is called at every step of training.
    """
    def __call__(self, *args: Any, step: int, **kwargs: Any) -> Any:
        ...


class BaseMonitor:
    """Base class for monitoring during training loops.
    
    Attributes:
        steps (Set[int]): The steps at which the callbacks should be called.
        callbacks (List[Callable]): List of callback functions to be executed.
        
    Methods:
        __call__: Executes the callbacks if the current step is in `steps`.
    """
    # ...
    steps: Set[int]
    callbacks: List[Callable]

    def __init__(self, steps: Union[List[int], Tuple[int], Set[int], StepsConfig], callbacks: List[Callable]):
        self.steps = process_steps(steps)
        self.callbacks = callbacks

    def __call__(self, *args: Any, step: int, **kwargs: Any) -> Any:
        if step in self.steps:
            for callback in self.callbacks:
                return callback(*args, **kwargs)
            

class CombineMonitors:
    """Combines multiple Monitor objects into a single callable.
    
    Attributes:
        monitors (List[Monitor]): List of Monitor objects to be combined.
        
    Methods:
        __call__: Executes all monitors for the given step.
    """
    monitors: List[Monitor]

    def __init__(self, monitors: List[Monitor]):
        self.monitors = monitors

    def __call__(self, *args: Any, step: int, **kwargs: Any) -> Any:
        for monitor in self.monitors:
            monitor(*args, step=step, **kwargs)


def expand_steps_config_(
    config: Union[StepsConfigShortened, StepsConfig], num_steps: int
) -> StepsConfig:
    """Expands a shortened steps configuration into a full configuration.
    
    Args:
        config (Union[StepsConfigShortened, StepsConfig]): The steps configuration.
        num_steps (int): The total number of steps.
        
    Returns:
        StepsConfig: The expanded steps configuration.
    """
    if isinstance(config.get("log_space", None), int):
        config["log_space"] = [1, num_steps, config["log_space"]]
    if isinstance(config.get("linear_space", None), int):
        config["linear_space"] = [0, num_steps, config["linear_space"]]


def process_steps(config: Union[List[int], Tuple[int], Set[int], StepsConfig]):
    """Processes the steps configuration and returns a set of steps.
    
    Args:
        config (Union[List[int], Tuple[int], Set[int], StepsConfig]): The steps configuration.
        
    Returns:
        Set[int]: The set of steps.
        
    Raises:
        ValueError: If the configuration is invalid.
    """
    if isinstance(config, dict):
        result = set()
        log_args = config.get("log_space")
        lin_args = config.get("linear_space")

        if log_args is not None:
            _results = int_logspace(*log_args, return_type="set")

            log_subsample = config.get("log_subsample")
            if log_subsample:
                _results = set(sorted(list(_results))[::log_subsample])

            result |= _results

        if lin_args is not None:
            _results = int_linspace(*lin_args, return_type="set")

            linear_subsample = config.get("linear_subsample")
            if linear_subsample:
                _results = set(sorted(list(_resultss))[::linear_subsample])

            result |= _results

        return result
    elif isinstance(config, (list, tuple, set)):
        return set(config)
    else:
        raise ValueError(f"Invalid steps config: {config}")