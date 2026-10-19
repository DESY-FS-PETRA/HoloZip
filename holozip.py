"""Source-checkout CLI. Installed users can run `holozip` instead."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from holozip.app import main

if __name__ == "__main__":
    main()
