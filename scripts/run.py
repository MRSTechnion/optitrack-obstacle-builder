"""
Direct entry point — use this if the package is not installed:

    python scripts/run.py

Or after `pip install -e .`:

    obstacle-builder
"""

import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from obstacle_builder.gui.main_window import main


if __name__ == "__main__":
    main()