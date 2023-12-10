import os
from pathlib import Path

FIGURES = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "figures"
ANALYSIS = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "analysis"
SWEEPS = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "sweeps"

