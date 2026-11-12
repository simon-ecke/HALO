"""Regression test: the statistics of core.stats reproduce the numbers of the old results notebook.

The golden values (tests/golden/site155.json) were read from the notebook's own
output tables, never from core code. Counts must match exactly; floats match to
the tolerance stored with each value (full precision where the old table stores
full precision, half a unit of the last printed digit where it is rounded).
Bootstrap intervals use the notebook's seed and resample count and match as well.
"""

import json
import math
from pathlib import Path

import pytest

from core import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs' / 'site155_all' / 'tables'
GOLDEN_FILE = ROOT / 'tests' / 'golden' / 'site155.json'
if not GOLDEN_FILE.is_file():
    pytest.skip('reference values (tests/golden/site155.json) are not distributed with the code', allow_module_level=True)
GOLDEN = json.loads(GOLDEN_FILE.read_text(encoding='utf-8'))['values']

# Settings of the site (config/site_155.yaml); passed as arguments, core.stats has no defaults for them
PARAMS = dict(high_confidence_distance_m=1.5, max_match_distance_m=3.0, gone_above=0.8, standing_below=0.2,
              dbh_plausible_mm=[70, 1000], standing_height_dm_above=100, top_found_distance_m=2.0, ellipse_axis_ratio=1.5)


@pytest.fixture(scope='module')
def entries():
    computed, _ = stats.compute_all(stats.load_tables(TABLES), **PARAMS)
    return {(e['id'], e['subset']): e for e in computed}


def test_golden_file_is_not_empty():
    assert len(GOLDEN) > 150


def test_entry_shape(entries):
    keys = {'id', 'value', 'unit', 'n', 'ci_low', 'ci_high', 'subset', 'statement'}
    assert len(entries) > 150
    for e in entries.values():
        assert set(e) == keys
        assert isinstance(e['statement'], str) and e['statement'].strip().endswith('.')


def test_accuracy_entries_cover_references_and_subsets(entries):
    for ref in ('massband', 'vertex', 'mean'):
        for subset in stats.SUBSETS:
            for measure in ('bias', 'rmse', 'median', 'mad'):
                e = entries[(f'height.{ref}.{measure}', subset)]
                assert e['value'] is not None and e['ci_low'] is not None and e['ci_high'] is not None
    for group in ('sample_trees', 'other_harvested_trees'):
        assert (f'dbh.{group}.bias', 'detected_only') in entries


@pytest.mark.parametrize('g', GOLDEN, ids=lambda g: f"{g['id']}|{g['subset']}|{g['field']}")
def test_golden(entries, g):
    got = entries[(g['id'], g['subset'])][g['field']]
    assert got is not None, 'statistic missing'
    if g['kind'] == 'count':
        assert got == g['expected'], f"{g['source']}"
    else:
        assert math.isclose(got, g['expected'], rel_tol=g['tol'].get('rel', 0), abs_tol=g['tol'].get('abs', 0)), \
            f"old {g['expected']!r} new {got!r} ({g['source']})"
