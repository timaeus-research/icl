"""
training the transformer on synthetic in-context regression task
"""
import torch
# manage environment
from dotenv import load_dotenv

from icl.evals import ICLEvaluator
from icl.utils import set_seed

load_dotenv()
# in case using mps:
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1" # (! before import torch)

import logging
from typing import Dict, List, Optional, TypedDict

import numpy as np
import sentry_sdk
import torch.nn.functional as F
import tqdm
#
from devinterp.optim.schedulers import LRScheduler

import wandb
from icl.config import ICLConfig, get_config
from icl.model import InContextRegressionTransformer
from icl.train import train


if __name__ == "__main__":
    sentry_sdk.init(
        dsn="https://92ea29f1e366cda4681fb10273e6c2a7@o4505805155074048.ingest.sentry.io/4505805162479616",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )

    logging.basicConfig(level=logging.INFO)
    config = get_config(project="icl", entity="devinterp")
    # config = get_config()
    train(config, is_debug=False)

