import os
from pathlib import Path

p = os.path.abspath(__file__)

ROOT_DIR = os.path.dirname(Path(p).parent.parent)
