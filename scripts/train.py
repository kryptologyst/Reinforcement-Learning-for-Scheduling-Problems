"""Training script entry point."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train.trainer import main

if __name__ == "__main__":
    main()
