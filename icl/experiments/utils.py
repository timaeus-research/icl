import logging
import os
from pathlib import Path

import devinfra
import seaborn as sns
import sentry_sdk
from dotenv import load_dotenv
from tqdm import tqdm

from icl.constants import FIGURES, ANALYSIS
from icl.setup import DEVICE

K=3  # Num cov components


def iter_models(model, checkpointer, verbose=False):
    for file_id in tqdm(checkpointer.file_ids, desc="Iterating over checkpoints", disable=not verbose):
        model.load_state_dict(checkpointer.load_file(file_id)["model"])
        yield model


def prepare_experiments():
    load_dotenv()
    sns.set_theme(style="whitegrid")

    assert os.path.exists(FIGURES)
    assert os.path.exists(ANALYSIS)

    logging.basicConfig(level=logging.INFO)
    # set_start_method('spawn')  # Required for sharing CUDA tensors
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


