import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SITE_155 = ROOT / "config" / "site_155.yaml"


@pytest.fixture
def cfg155():
    from core.config import load_config

    return load_config(SITE_155)


@pytest.fixture
def site155_path():
    return SITE_155
