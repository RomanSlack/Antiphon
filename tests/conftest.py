import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

REF_PATH = Path(__file__).resolve().parent.parent / 'refs' / 'urban_anc_simulation.py'


@pytest.fixture(scope='session')
def ref_sim():
    """The original monolithic simulation script, loaded as a module."""
    spec = importlib.util.spec_from_file_location('ref_sim', REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
