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
TABS = ["overview", "data", "match", "height", "wins", "samples", "volume", "methods"]
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
