import copy
from pathlib import Path

import pytest
import yaml

from core.config import ConfigError, load_config
from core.validate import validate


def _write(tmp_path, raw):
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture
def raw155(site155_path):
    return yaml.safe_load(site155_path.read_text(encoding="utf-8"))


@pytest.fixture
def need_data(raw155):
    if not Path(raw155["paths"]["plot_corners"]).is_file():
        pytest.skip("input data on D: not reachable")


def test_loads_and_exposes(cfg155):
    assert cfg155.matching.max_distance_m == 3.0
    assert cfg155.paths.sat_before.path.is_absolute()
    assert cfg155.output_dir.is_absolute() and cfg155.output_dir.name == "site155"
    assert cfg155.table("x.csv") == cfg155.tables_dir / "x.csv"
    assert cfg155.baseline.windows[4].per_height_m == 0.08
    assert cfg155.section_hash("matching") != cfg155.section_hash("grid")
    assert isinstance(cfg155.raw, dict)


def test_site155_validates(site155_path, need_data):
    assert validate(site155_path) == []


def test_missing_required_key(tmp_path, raw155):
    raw = copy.deepcopy(raw155)
    del raw["matching"]["max_distance_m"]
    path = _write(tmp_path, raw)
    with pytest.raises(ConfigError, match=r"matching\.max_distance_m"):
        load_config(path)
    assert any("matching.max_distance_m" in m for m in validate(path))


def test_missing_input_file(tmp_path, raw155, need_data):
    raw = copy.deepcopy(raw155)
    raw["paths"]["sat_before"]["path"] = str(tmp_path / "nope.laz")
    msgs = validate(_write(tmp_path, raw))
    assert any(m.startswith("paths.sat_before.path: file not found") for m in msgs)


def test_wrong_column_mapping(tmp_path, raw155, need_data):
    raw = copy.deepcopy(raw155)
    raw["field_columns"]["layouts"][1]["columns"]["dbh_mm"] = "NoSuchColumn"
    msgs = validate(_write(tmp_path, raw))
    assert any("field_columns.layouts[1].columns.dbh_mm" in m and "NoSuchColumn" in m for m in msgs)
