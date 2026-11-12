"""Table cache keyed on config sections and input files.

cached_table() returns a DataFrame. The table is stored as CSV at cfg.table(name); the key
(section hash plus size and mtime of every input file) is stored in <output_dir>/cache/<name>.json.
A table read back from the cache has the dtypes of pandas.read_csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_FORCE = False


def set_force(value: bool = True) -> None:
    """Make every cached_table() call recompute (used by --force)."""
    global _FORCE
    _FORCE = bool(value)


def _file_state(path) -> list:
    p = Path(path)
    if not p.exists():
        return [str(p), None, None]
    st = p.stat()
    return [str(p), st.st_size, st.st_mtime_ns]


def _key(cfg, depends_on_sections, input_files) -> dict:
    return {"sections": list(depends_on_sections),
            "section_hash": cfg.section_hash(*depends_on_sections),
            "inputs": [_file_state(f) for f in input_files]}


def cached_table(cfg, name, depends_on_sections, input_files, compute) -> pd.DataFrame:
    """Return the table `name`, computing it with compute() only when needed.

    Recomputes when the hash of the named config sections or the size/mtime of an input file
    changed, when the table or its key file is missing, or when set_force(True) was called.
    """
    table_path = cfg.table(name)
    key_path = cfg.cache_dir / f"{name}.json"
    key = _key(cfg, depends_on_sections, input_files)

    if not _FORCE and table_path.is_file() and key_path.is_file():
        try:
            stored = json.loads(key_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = None
        if stored == key:
            return pd.read_csv(table_path)

    df = compute()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False, encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
    return df
