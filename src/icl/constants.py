import logging
import os
from pathlib import Path

from icl.monitoring import LOG_FILE_NAME, LOGGING_LEVEL
from icl.utils import get_default_device

PROJECT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES = PROJECT_DIR / "figures"
ANALYSIS = PROJECT_DIR / "analysis"
SWEEPS = PROJECT_DIR / "sweeps"
DATA = PROJECT_DIR / "data"

if not os.path.exists(FIGURES):
    os.makedirs(FIGURES)


if not os.path.exists(ANALYSIS):
    os.makedirs(ANALYSIS)

DEVICE = get_default_device()
XLA = DEVICE.type == "xla"

# Language

LANGUAGE_FILEPATH = DATA / "train-5m.jsonl"
UNIGRAMS_FILEPATH = DATA / "unigram_freq_percents.pkl"
BIGRAMS_FILEPATH = DATA / "bigram_freq_percents.pkl"
TRIGRAMS_FILEPATH = DATA / "trigram_freq_percents.pkl"