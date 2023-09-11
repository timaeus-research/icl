"""
training the transformer on synthetic in-context regression task
"""
import logging

from dotenv import load_dotenv
import sentry_sdk

from icl.config import get_config
from icl.train import train

load_dotenv()


if __name__ == "__main__":
    import torch_xla.debug.profiler as xp
    server = xp.start_server(9012)

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
    # config = get_config(project="icl", entity="devinterp")
    config = get_config()
    train(config, is_debug=False)

