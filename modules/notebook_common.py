"""Paths, settings and plot style shared by the Versuch 155 notebooks.

Both notebooks import everything from here, so a threshold or a path is set in
one place and the pipeline and the results notebook cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _data_root() -> Path:
    """Input data root: environment variable HALO_DATA_ROOT, else data_root.local in the project root."""
    value = os.environ.get('HALO_DATA_ROOT', '').strip()
    local = PROJECT_ROOT / 'data_root.local'
    if not value and local.is_file():
        value = local.read_text(encoding='utf-8').strip().splitlines()[0].strip()
    if not value:
        raise RuntimeError('Set HALO_DATA_ROOT or write the data folder into data_root.local (see README).')
    return Path(value)


INPUT_ROOT = _data_root()
FIELD_DIR = INPUT_ROOT / '00_field_data_tree_heights'
EXCEL_PATHS = [FIELD_DIR / '155_EBEe_mitVertexund Maßband.xlsx',      # Parzellen 1-26
               FIELD_DIR / '155_EBEe-P27-45_done_ausg 1.xlsx']        # Parzellen 27-45
CORNERS_PATH = FIELD_DIR / 'Parzelleneckpunkte_155.xlsx'

DATA_ROOT = INPUT_ROOT / '03_Data_output'
SAT_DIR = DATA_ROOT / '03_SegmentAnyTree'
# Before-harvest flights: SegmentAnyTree file, name of its tree-id field, output folder
BEFORE_FLIGHTS = {
    '20260225': (SAT_DIR / '20260225_EBE-e_155_L3_80H80O8ms_nonrep_SAT.laz', 'PredInstance', 'site155_all'),
    '20260318': (SAT_DIR / '20260318_EBE-e_155_L3_80H80O8ms_nonrep_double_SAT.laz', 'PredInstance_SAT',
                 'site155_all_20260318_double'),
}
SAT_AFTER = SAT_DIR / '20260324_EBE-e_155_L3_80H80O8ms_nonrep_SAT.laz'
DSM_BEFORE = DATA_ROOT / '02_Photogrammetrie' / 'DSMs' / '20260225_EBEeiche_M3E_DSM.tif'
DSM_AFTER = DATA_ROOT / '02_Photogrammetrie' / 'DSMs' / '20260324_EBEeiche_M3E_DSM.tif'
ORTHO_BEFORE = DATA_ROOT / '02_Photogrammetrie' / 'Orthos' / '20260225_EBEeiche_M3E_Ortho.tif'
ORTHO_AFTER = DATA_ROOT / '02_Photogrammetrie' / 'Orthos' / '20260324_EBEeiche_M3E_Ortho.tif'

CRS = 'EPSG:25832'


def output_dir(before_flight: str = '20260225') -> Path:
    """Output folder of one before-harvest flight."""
    return PROJECT_ROOT / 'outputs' / BEFORE_FLIGHTS[before_flight][2]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

AOI_BUFFER_M = 20.0
DTM_RESOLUTION_M = 0.5
CHM_RESOLUTION_M = 0.25
REMOVED_VOXEL_M = 0.4               # voxel size for "is this point gone after harvest"

# Parzellen whose local y in the sheet runs from the opposite plot edge (y -> 30 - y)
Y_MIRRORED_PARZELLEN = [41, 42, 43, 44, 45]

# Matching
MAX_MATCH_DISTANCE_M = 3.0          # stem -> instance base
REMOVED_THRESHOLD = 0.5             # instance counts as felled above this removed fraction
REMOVAL_PENALTY_M = 2.0             # cost added when harvest evidence disagrees
HIGH_CONFIDENCE_DISTANCE_M = 1.5    # 'detected' needs the base within this distance
HEIGHT_GATE_REL = 0.25              # flag only: |LiDAR - field| / field
CONTROL_SHIFT_M = 5.0               # chance control: stems shifted by this distance
GONE_ABOVE, STANDING_BELOW = 0.8, 0.2   # share of an instance's points gone after the harvest: felled above, standing below

# Field values
DBH_PLAUSIBLE_MM = (70, 1000)       # values outside are treated as typing errors
STANDING_HEIGHT_DM_ABOVE = 100      # standing heights above this value were typed in dm (e.g. 316 = 31.6 m)

# LiDAR DBH
DBH_BREAST_HEIGHT_M = 1.3
DBH_SLICE_HALF_M = 0.25             # slice from 1.05 to 1.55 m
DBH_SEARCH_RADIUS_M = 1.0
ELLIPSE_MAX_AXIS_RATIO = 1.5        # limit set in ftm.fit_ellipse

# Photogrammetry
PHOTO_RADIUS_M = 1.0                # nDSM max within this radius of the LiDAR top
PHOTO_DETECT_DISTANCE_M = 1.5       # photo top counts as the same tree within this distance
VISIBLE_TOLERANCE_M = 1.0           # top may lie this far below the canopy surface

# Stand structure
COMPETITION_RADIUS_M = 5.0          # neighbours within this distance count for the competition index
SMALL_DBH_MM = 250                  # "small tree" for the per-Parzelle share
CLOSE_NEIGHBOUR_M = 2.0
NO_CANOPY_DEFICIT_M = 6.0           # canopy at the stem this far below the expected tree height: no full-size tree there
CHM_STEM_RADIUS_M = 1.0
EDGE_M = 2.0                        # a tree closer than this to the Parzelle outline counts as an edge tree
MIN_STEM_SLICES = 3                 # stem ownership check: stem found in at least this many of 8 slices

# Matching with DBH and height (algorithm comparison)
FOUND_DISTANCE_M = 1.5
DBH_TOLERANCE_MM = 100              # LiDAR and field DBH may differ this much at no cost
DBH_PENALTY_M = 1.0                 # a larger difference costs as much as this extra distance
HEIGHT_TOLERANCE_M = 5.0            # only an obvious height difference counts
HEIGHT_PENALTY_M = 1.0
ALGORITHMS = {
    '1 position': dict(),
    '2 position + DBH': dict(dbh_tolerance_mm=DBH_TOLERANCE_MM, dbh_penalty_m=DBH_PENALTY_M),
    '3 position + DBH + height': dict(dbh_tolerance_mm=DBH_TOLERANCE_MM, dbh_penalty_m=DBH_PENALTY_M,
                                      height_tolerance_m=HEIGHT_TOLERANCE_M, height_penalty_m=HEIGHT_PENALTY_M),
}

# Baseline: local maxima on the LiDAR CHM, matched to the field stems by their tops
BASELINE_WINDOWS = {                # window diameter in m, fixed or as a function of the cell height
    'LM 1.5 m': 1.5, 'LM 2 m': 2.0, 'LM 3 m': 3.0, 'LM 4 m': 4.0,
    'LM 1.5 m + 0.08 h': lambda h: 1.5 + 0.08 * h,
}
BASELINE_MIN_HEIGHT_M = 3.0         # as the smallest SegmentAnyTree instance that is kept
BASELINE_SMOOTHING_M = 0.25
TOP_MATCH_MAX_M = 3.0               # stem -> top, largest distance for the assignment
TOP_FOUND_DISTANCE_M = 2.0          # a field tree counts as found if its top lies within this distance of the stem
                                    # (stem and top of a leaning tree are apart: 95 % of the detected trees within 2.1 m)

# Stem-seeded refinement, REFINE_TREES = 'auto': every sample tree, plus standing-height trees with a larger error
REFINE_OUTLIER_M = 3.0

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

CLASSES = ['detected', 'uncertain', 'not detected']
CLASS_COLORS = {'detected': '#2a78d6', 'uncertain': '#eb6834', 'not detected': '#898781'}
LIDAR_ONLY_COLOR = '#1baf7a'
ALGO_COLORS = dict(zip(ALGORITHMS, ['#2a78d6', '#eb6834', '#1baf7a']))
INK, INK_SECONDARY, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
PAPER = '#fcfcfb'

SCREEN_RC = {
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK_SECONDARY, 'text.color': INK,
    'xtick.color': INK_SECONDARY, 'ytick.color': INK_SECONDARY, 'grid.color': GRID,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'legend.frameon': False,
}

# Publication figures (white background, small type, embedded fonts)
PUB_LIDAR, PUB_PHOTO, PUB_PAIR, PUB_UNCERTAIN = '#2a78d6', '#eb6834', '#3b3b39', '#898781'
PUB_RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8.5,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.edgecolor': INK_SECONDARY, 'axes.linewidth': 0.7, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK_SECONDARY, 'ytick.color': INK_SECONDARY, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
}


def apply_screen_style() -> None:
    """Figure style of the working figures (one colour per detection class)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(SCREEN_RC)
