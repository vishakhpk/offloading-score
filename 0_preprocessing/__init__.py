"""Initialize the 0_preprocessing package by adding parent directory to sys.path."""

import sys
from pathlib import Path

# Add parent directory to path for utils and language imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


def _setup_path():
    """Helper function to set up path (for scripts that import this module)."""
    pass  # Path is already set up when this module is imported

