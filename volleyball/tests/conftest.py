import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# optional local checkout of py-datavolley (pure Python) for machines without PyPI access
if os.environ.get("PYDATAVOLLEY_SRC"):
    sys.path.insert(0, os.environ["PYDATAVOLLEY_SRC"])

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DV = importlib.util.find_spec("datavolley") is not None
