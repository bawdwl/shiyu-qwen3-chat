"""Run existing Python regression tests without loading the real GPU model."""
import os
import sys
import unittest
from project_paths import configure


if __name__ == '__main__':
    root = configure()
    os.chdir(root)
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
