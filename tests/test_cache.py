import copy

import pandas as pd
import pytest

from core import cache
from core.config import Config


@pytest.fixture
def cfg(cfg155, tmp_path):
    raw = copy.deepcopy(cfg155.raw)
    raw["paths"]["output_dir"] = str(tmp_path / "out")
    return Config(raw)


@pytest.fixture(autouse=True)
def _no_force():
    cache.set_force(False)
    yield
    cache.set_force(False)


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return pd.DataFrame({"a": [1, 2, 3]})


def run(cfg, c, inputs=()):
    return cache.cached_table(cfg, "t.csv", ["matching"], inputs, c)


def test_no_recompute_without_change(cfg, tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("a")
    c = Counter()
    run(cfg, c, [f])
    df = run(cfg, c, [f])
    assert c.n == 1 and df["a"].tolist() == [1, 2, 3]
    assert (cfg.cache_dir / "t.csv.json").is_file()


def test_recompute_on_section_change(cfg, tmp_path):
    c = Counter()
    run(cfg, c)
    raw = copy.deepcopy(cfg.raw)
    raw["matching"]["max_distance_m"] = 2.5
    run(Config(raw), c)
    assert c.n == 2
    raw["grid"]["aoi_buffer_m"] = 99.0  # section not listed as a dependency
    run(Config(raw), c)
    assert c.n == 2


def test_recompute_on_input_change(cfg, tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("a")
    c = Counter()
    run(cfg, c, [f])
    f.write_text("longer content")
    run(cfg, c, [f])
    assert c.n == 2


def test_force(cfg):
    c = Counter()
    run(cfg, c)
    cache.set_force(True)
    run(cfg, c)
    assert c.n == 2
