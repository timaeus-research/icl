import logging
import os
from pathlib import Path

from icl.monitoring import LOG_FILE_NAME, LOGGING_LEVEL
from icl.utils import get_default_device

# Paths

PROJECT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES = PROJECT_DIR / "figures"
SWEEPS = PROJECT_DIR / "sweeps"
DATA = PROJECT_DIR / "data"

# Language

LANGUAGE_DATA = DATA / "language"
LANGUAGE_FILEPATH = LANGUAGE_DATA / "train-5m.jsonl"
UNIGRAMS_FILEPATH = LANGUAGE_DATA / "unigram_freq_percents.pkl"
BIGRAMS_FILEPATH = LANGUAGE_DATA / "bigram_freq_percents.pkl"
TRIGRAMS_FILEPATH = LANGUAGE_DATA / "trigram_freq_percents.pkl"

# Create directories if they don't exist

for d in [FIGURES, SWEEPS, DATA, LANGUAGE_DATA]:
    if not os.path.exists(d):
        os.makedirs(d)

# General purpose
        
DEVICE = get_default_device()
XLA = DEVICE.type == "xla"
