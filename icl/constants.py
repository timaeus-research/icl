import os
from pathlib import Path

from icl.utils import get_default_device

FIGURES = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "figures"
ANALYSIS = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "analysis"
SWEEPS = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "sweeps"


if not os.path.exists(FIGURES):
    os.makedirs(FIGURES)


if not os.path.exists(ANALYSIS):
    os.makedirs(ANALYSIS)

DEVICE = get_default_device()
XLA = DEVICE.type == "xla"