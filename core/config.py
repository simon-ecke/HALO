"""Site configuration: load a YAML file, validate it against config/schema.json, expose it.

    cfg = load_config("config/site_155.yaml")
    cfg.matching.max_distance_m      # attribute access for sections
    cfg.paths.sat_before.path        # absolute pathlib.Path
    cfg.table("field_trees.csv")     # <output_dir>/tables/field_trees.csv
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "config" / "schema.json"

# Keys under "paths" that hold a file or folder, and keys holding {path: ..., instance_dim: ...}
_PATH_KEYS = ("output_dir", "sample_heights_table", "plot_corners", "ortho_before", "match_overrides",
              "manual_segments_dir")
_LAZ_KEYS = ("sat_before", "sat_after")


class ConfigError(Exception):
    """The configuration file is missing, unreadable or invalid. The message is plain text."""


class Section:
    """Read-only attribute view of a dict (nested dicts and lists of dicts are wrapped too)."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            return _wrap(self._data[name])
        except KeyError:
            raise AttributeError(f"config has no key '{name}'") from None

    def __setattr__(self, name, value):
        raise AttributeError("config sections are read-only")

    def __getitem__(self, name):
        return _wrap(self._data[name])

    def __contains__(self, name):
        return name in self._data

    def get(self, name, default=None):
        return _wrap(self._data[name]) if name in self._data else default

    def keys(self):
        return self._data.keys()

    def items(self):
        return [(k, _wrap(v)) for k, v in self._data.items()]

    def to_dict(self) -> dict:
        return self._data

    def __repr__(self):
        return f"Section({self._data!r})"


def _wrap(value):
    if isinstance(value, dict):
        return Section(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


DATA_ROOT_VAR = "HALO_DATA_ROOT"
DATA_ROOT_FILE = PROJECT_ROOT / "data_root.local"


def data_root() -> str | None:
    """Root folder of the input data: environment variable HALO_DATA_ROOT, else the first line of
    data_root.local in the project root (not under version control). None if neither is set."""
    import os
    value = os.environ.get(DATA_ROOT_VAR, "").strip()
    if not value and DATA_ROOT_FILE.is_file():
        lines = DATA_ROOT_FILE.read_text(encoding="utf-8").strip().splitlines()
        value = lines[0].strip() if lines else ""
    return value.rstrip("\\/") or None


def _expand(value):
    """Replace ${HALO_DATA_ROOT} in every string of a parsed YAML document. Without a data root the
    placeholder stays, so the configuration still loads and a missing input file names the variable."""
    token = "${" + DATA_ROOT_VAR + "}"
    if isinstance(value, str):
        root = data_root() if token in value else None
        return value.replace(token, root) if root else value
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _abs(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _resolve_paths(raw: dict) -> dict:
    """Copy of raw in which every entry of 'paths' is an absolute Path."""
    out = copy.deepcopy(raw)
    paths = out.get("paths")
    if not isinstance(paths, dict):
        return out
    for key in _PATH_KEYS:
        if isinstance(paths.get(key), str):
            paths[key] = _abs(paths[key])
    if isinstance(paths.get("field_tables"), list):
        paths["field_tables"] = [_abs(p) for p in paths["field_tables"] if isinstance(p, str)]
    for key in _LAZ_KEYS:
        if isinstance(paths.get(key), dict) and isinstance(paths[key].get("path"), str):
            paths[key]["path"] = _abs(paths[key]["path"])
    return out


def _fmt_path(parts) -> str:
    text = ""
    for p in parts:
        text += f"[{p}]" if isinstance(p, int) else (f".{p}" if text else str(p))
    return text


def schema_problems(raw) -> list[str]:
    """Return one plain line per schema violation, each starting with the dotted key."""
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    problems = []
    for err in sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda e: list(map(str, e.absolute_path))):
        where = _fmt_path(err.absolute_path)
        msg = err.message
        if err.validator in ("required", "additionalProperties"):
            names = re.findall(r"'([^']+)'", msg)
            if err.validator == "required" and names:
                where = (where + "." if where else "") + names[0]
                msg = "required key is missing"
            elif names:
                where = (where + "." if where else "") + names[0]
                msg = "unknown key"
        elif err.validator == "propertyNames":
            names = re.findall(r"'([^']+)'", msg)
            if names:
                where = (where + "." if where else "") + names[0]
                msg = "not a known standard name"
        elif len(msg) > 120:
            msg = msg[:117] + "..."
        problems.append(f"{where or '(top level)'}: {msg}")
    return problems


def read_raw(path) -> dict:
    """Read the YAML file into a dict. Raises ConfigError if it cannot be read."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the file must contain a mapping of top-level keys")
    return _expand(raw)


class Config:
    """Validated configuration of one site. Build with load_config()."""

    def __init__(self, raw: dict):
        self.raw = raw
        self._view = Section(_resolve_paths(raw))
        self.output_dir: Path = self._view.paths.output_dir
        self.tables_dir: Path = self.output_dir / "tables"
        self.cache_dir: Path = self.output_dir / "cache"
        self.project_root: Path = PROJECT_ROOT

    def __getattr__(self, name):
        # Only called when normal lookup fails: forward section names to the view.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._view, name)

    def table(self, name: str) -> Path:
        """Path of a table file in <output_dir>/tables."""
        return self.tables_dir / name

    def section_hash(self, *keys: str) -> str:
        """Hash of the named top-level sections (as written in the file); changes when any value in them changes."""
        part = {k: self.raw.get(k) for k in sorted(keys)}
        text = json.dumps(part, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path) -> Config:
    """Load and validate a site config. Raises ConfigError listing every problem."""
    raw = read_raw(path)
    problems = schema_problems(raw)
    if problems:
        raise ConfigError(f"{Path(path).name} is not valid:\n" + "\n".join(problems))
    return Config(raw)
