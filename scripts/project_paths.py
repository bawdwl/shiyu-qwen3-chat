"""Local entry points share the product and experiment import paths."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure():
    for path in (ROOT, ROOT / 'experiments' / 'lora', ROOT / 'src'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return ROOT
