"""Build the static dashboard: python scripts/build_dashboard.py <config> [--tables-dir DIR] [--out DIR]

Writes dist/<site_id>/ with index.html, app.js, style.css, data.js, plotly.min.js and img/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from core.export import build_data, write_data_js  # noqa: E402


def plotly_js_path() -> Path:
    import plotly

    p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not p.exists():
        raise FileNotFoundError(f"plotly.min.js not found in the installed plotly package: {p}")
    return p


def build(config_path, tables_dir=None, out_dir=None) -> Path:
    cfg = load_config(config_path)
    out = Path(out_dir) if out_dir else ROOT / "dist" / str(cfg.site.id)
    out.mkdir(parents=True, exist_ok=True)
    data, figs = build_data(cfg, tables_dir, out)
    write_data_js(data, out / "data.js")
    for name in ("app.js", "style.css"):
        shutil.copyfile(ROOT / "app" / name, out / name)
    shutil.copyfile(plotly_js_path(), out / "plotly.min.js")
    # Version the file references so that a browser never mixes a new page with cached scripts or data
    stamp = hashlib.sha256(b"".join((out / n).read_bytes() for n in ("data.js", "app.js", "style.css"))).hexdigest()[:10]
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    for name in ("style.css", "data.js", "app.js"):
        html = html.replace(f'"{name}"', f'"{name}?v={stamp}"')
    (out / "index.html").write_text(html, encoding="utf-8")
    if figs:
        img = out / "img"
        img.mkdir(exist_ok=True)
        sf = cfg.dashboard.sample_figures
        src = Path(sf.dir)
        src = src if src.is_absolute() else cfg.project_root / src
        for name in figs:
            shutil.copyfile(src / name, img / name)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--tables-dir", default=None, help="read a table from this folder if it exists there, else from the site's tables folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = build(a.config, a.tables_dir, a.out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"dashboard written to {out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
