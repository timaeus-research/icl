"""
training the transformer on synthetic in-context regression task
"""

import dotenv; dotenv.load_dotenv() # environment before all imports

import logging

import sentry_sdk

from icl.config import get_config
from icl.train import train


if __name__ == "__main__":
    sentry_sdk.init(
        dsn="https://92ea29f1e366cda4681fb10273e6c2a7@o4505805155074048.ingest.sentry.io/4505805162479616",
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

    logging.basicConfig(level=logging.INFO)
    config = get_config(project="icl", entity="devinterp")
    # config = get_config()
    train(config, is_debug=False)

