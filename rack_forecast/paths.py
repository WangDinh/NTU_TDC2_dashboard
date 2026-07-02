"""Filesystem roots for the project.

All paths are resolved from THIS file's location (`__file__`), not the current
working directory. That means notebooks, CLI scripts, and the Streamlit
dashboard all agree on where `data/` and `results/` live no matter where they
are launched from — the single most common source of "works here, breaks there"
bugs in research code.
"""

from pathlib import Path

# rack_forecast/paths.py  →  parents[1] is the project root (the NTU/ folder).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT    = PROJECT_ROOT / 'data' / 'TDC2.0 Dataset'   # raw TDC 2.0 dataset
RESULTS_ROOT = PROJECT_ROOT / 'results'                    # experiment outputs

# Make sure results/ exists so first-run saves never fail.
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
