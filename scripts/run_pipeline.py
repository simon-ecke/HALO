"""Run the pipeline stages of one site.

    python scripts\run_pipeline.py config\site_155.yaml [--stage NAME] [--force]

Stages run in the order below. Each module is imported only when its stage is called, so a stage whose module
is not written yet fails only when it is run. A stage calls run_<stage>(cfg) if the module has it, else run(cfg).
Excel is read by the inventory stage, which comes first (Excel must be read before geopandas or rasterio).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = [
    ("inventory", "core.io_field"),
    ("instances", "core.instances"),
    ("orientation", "core.coords"),
    ("matching", "core.matching"),
    ("structure", "core.causes"),
    ("stems", "core.dbh"),
    ("causes", "core.causes"),
    ("dbh", "core.dbh"),
    ("tiers", "core.tiers"),
    ("unmatched", "core.matching"),
    ("baseline", "core.baseline"),
    ("samples", "core.samples"),
    ("identity", "core.identity"),
    ("volume", "core.volume"),
]


def run_stage(name: str, module_name: str, cfg) -> None:
    module = importlib.import_module(module_name)
    func = getattr(module, f"run_{name}", None) or getattr(module, "run", None)
    if func is None:
        raise AttributeError(f"{module_name} has neither run_{name}(cfg) nor run(cfg)")
    func(cfg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", help="path of the site YAML file")
    parser.add_argument("--stage", choices=[s for s, _ in STAGES], help="run only this stage")
    parser.add_argument("--force", action="store_true", help="recompute cached tables")
    args = parser.parse_args(argv)

    from core import cache
    from core.config import load_config

    cfg = load_config(args.config)
    cache.set_force(args.force)
    stages = [(s, m) for s, m in STAGES if args.stage in (None, s)]
    for name, module_name in stages:
        t0 = time.time()
        try:
            run_stage(name, module_name, cfg)
        except Exception as exc:                          # report and stop: later stages need this one
            print(f"[{name}] FAILED after {time.time() - t0:.1f} s: {type(exc).__name__}: {exc}")
            return 1
        print(f"[{name}] done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
