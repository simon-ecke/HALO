"""Build the dashboard and open every tab in headless Edge; no JS errors, something drawn per tab."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
TABS = ["overview", "data", "match", "height", "wins", "samples", "volume", "volume_gri", "methods"]
OLD_TABLES = ROOT / "outputs" / "site155_all" / "tables"


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_dashboard import build

    out = tmp_path_factory.mktemp("dist") / "155"
    if not OLD_TABLES.exists() and not (ROOT / "outputs" / "app" / "site155" / "tables").exists():
        pytest.skip("pipeline tables of site 155 are not available")
    build(ROOT / "config" / "site_155.yaml", OLD_TABLES if OLD_TABLES.exists() else None, out)
    return out


def _dump(index: Path, tab: str, profile: Path) -> str:
    url = index.as_uri() + "#tab=" + tab
    r = subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--virtual-time-budget=8000",
                        f"--user-data-dir={profile}", "--dump-dom", url], capture_output=True, timeout=180)
    return r.stdout.decode("utf-8", "replace")


@pytest.mark.slow
@pytest.mark.skipif(not EDGE.exists(), reason="Edge is not installed")
@pytest.mark.parametrize("tab", TABS)
def test_tab_renders(dist, tab):
    with tempfile.TemporaryDirectory() as prof:
        dom = _dump(dist / "index.html", tab, Path(prof))
    body = re.search(r"<body[^>]*>", dom).group(0)
    attr = lambda k: re.search(k + r'="([^"]*)"', body).group(1)  # noqa: E731
    errors = json.loads(attr("data-errors").replace("&quot;", '"'))
    assert errors == [], errors
    assert attr("data-tab") == tab
    assert int(attr("data-rendered")) + int(attr("data-tables")) >= 1
    if tab not in ("methods", "data"):
        assert int(attr("data-rendered")) >= 1


@pytest.mark.slow
@pytest.mark.skipif(not EDGE.exists(), reason="Edge is not installed")
def test_volume_gri_consistent(dist):
    """The browser calculation of the GRI volumes reproduces the Python stand totals."""
    data = (dist / "data.js").read_text(encoding="utf-8")
    if '"volume_gri":{"available":true' not in data:
        pytest.skip("GRI volume tables are not available")
    with tempfile.TemporaryDirectory() as prof:
        dom = _dump(dist / "index.html", "volume_gri", Path(prof))
    m = re.search(r'data-gri-maxdiff="([^"]*)"', dom)
    assert m, "data-gri-maxdiff is missing"
    assert float(m.group(1)) < 0.001


@pytest.mark.slow
@pytest.mark.skipif(not EDGE.exists(), reason="Edge is not installed")
def test_volume_gri_increment_section(dist):
    """The increment section opens with an age and the comparison of all three methods from the page address."""
    data = (dist / "data.js").read_text(encoding="utf-8")
    if '"volume_gri":{"available":true' not in data or '"increment":{"available":true' not in data:
        pytest.skip("GRI volume or increment tables are not available")
    with tempfile.TemporaryDirectory() as prof:
        dom = _dump(dist / "index.html", "volume_gri&age=90&zm=all", Path(prof))
    body = re.search(r"<body[^>]*>", dom).group(0)
    errors = json.loads(re.search(r'data-errors="([^"]*)"', body).group(1).replace("&quot;", '"'))
    assert errors == [], errors
    assert 'class="err"' not in dom
    assert 'data-increment-age="90"' in dom
    m = re.search(r'data-increment-methods="([^"]*)"', dom)
    assert m and len(m.group(1).split(",")) == 3


def _increment_dom(dist, hash_tail):
    data = (dist / "data.js").read_text(encoding="utf-8")
    if '"volume_gri":{"available":true' not in data or '"increment":{"available":true' not in data:
        pytest.skip("GRI volume or increment tables are not available")
    with tempfile.TemporaryDirectory() as prof:
        return _dump(dist / "index.html", hash_tail, Path(prof)), data


@pytest.mark.slow
@pytest.mark.skipif(not EDGE.exists(), reason="Edge is not installed")
def test_volume_gri_measured_increment_matches_python(dist):
    """The browser recalculation of the measured increment with the default parameters equals the Python statistic."""
    dom, data = _increment_dom(dist, "volume_gri&zm=measured")
    m = re.search(r'data-increment-measured="([^"]*)"', dom)
    assert m and m.group(1), "data-increment-measured is missing"
    start = data.index('"increment.measured.stand.pai"')
    value = float(re.search(r'"value":\s*([-0-9.eE]+)', data[start:start + 400]).group(1))
    assert float(m.group(1)) == pytest.approx(value, rel=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(not EDGE.exists(), reason="Edge is not installed")
def test_volume_gri_measured_parameters_from_hash(dist):
    """Changed growing seasons and growth share from the page address render without errors."""
    dom, _ = _increment_dom(dist, "volume_gri&zm=measured&gs=2025:5,2026:6&rg=0.5")
    body = re.search(r"<body[^>]*>", dom).group(0)
    errors = json.loads(re.search(r'data-errors="([^"]*)"', body).group(1).replace("&quot;", '"'))
    assert errors == [], errors
    assert 'class="err"' not in dom
    assert 'data-ledger="measured"' in dom
