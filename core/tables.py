"""Read pipeline tables by name, with an optional override folder.

`tables_dir` lets tests and development runs read tables from another folder
(for example the tables of an earlier run). A table that is missing there is
read from the config's own tables folder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def table_path(cfg, name: str, tables_dir: Path | str | None = None) -> Path:
    """Return the path of table `name`: in `tables_dir` if it exists there, else in cfg.tables_dir."""
    if tables_dir is not None:
        candidate = Path(tables_dir) / name
        if candidate.exists():
            return candidate
    return Path(cfg.table(name))


def has_table(cfg, name: str, tables_dir: Path | str | None = None) -> bool:
    return table_path(cfg, name, tables_dir).exists()


def read_table(cfg, name: str, tables_dir: Path | str | None = None, **kwargs) -> pd.DataFrame:
    """Read a CSV table by name (see `table_path` for where it is looked up)."""
    path = table_path(cfg, name, tables_dir)
    if not path.exists():
        raise FileNotFoundError(f"Table '{name}' not found in {path.parent}. Run the stage that writes it first.")
    return pd.read_csv(path, **kwargs)
