import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\New folder (2)")
B_IDX, G_IDX, R_IDX = 24, 70, 95
TOL = 6.0  # if JPGs are used, you may need 6-12

# --- Helpers ----------------------------------------------------

def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    # Convert BGR->RGB for consistent comparison; we'll also try BGR later
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_cube(path: Path) -> np.ndarray:
    cube = np.load(str(path))
    if cube.ndim == 2:
        cube = cube[..., None]
    return cube

def clamp_band_idx(idx: int, C: int) -> int:
    return max(0, min(idx, C - 1))

def pseudo_from_roi_with_globals(cube: np.ndarray,
                                 global_min: dict,
                                 global_max: dict,
                                 b_idx=B_IDX, g_idx=G_IDX, r_idx=R_IDX) -> np.ndarray:
    """Normalize ROI bands using GLOBAL (per-image) min/max, then stack as RGB."""
    H, W, C = cube.shape
    b_idx = clamp_band_idx(b_idx, C)
    g_idx = clamp_band_idx(g_idx, C)
    r_idx = clamp_band_idx(r_idx, C)

    def norm_with_globals(band: np.ndarray, mn: float, mx: float) -> np.ndarray:
        if mx <= mn:
            return np.zeros_like(band, dtype=np.uint8)
        x = (band.astype(np.float32) - mn) / (mx - mn)
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
        return x

    blue  = norm_with_globals(cube[..., b_idx], global_min[b_idx], global_max[b_idx])
    green = norm_with_globals(cube[..., g_idx], global_min[g_idx], global_max[g_idx])
    red   = norm_with_globals(cube[..., r_idx], global_min[r_idx], global_max[r_idx])

    rgb = np.dstack([red, green, blue])  # true RGB
    return rgb

def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

# --- Grouping logic ---------------------------------------------

# Your filenames look like:
#  3_G31R2_2024-07-17_18-25-09_<class>_<timestamp>_<idx>.jpg/.npy
# We want the "base id" before the class part.
BASE_RE = re.compile(r"^(.*)_([A-Za-z0-9]+)_[0-9]{13}_[0-9]+$")

def base_id_of(stem: str) -> str:
    """
    Return the base source id (e.g., '3_G31R2_2024-07-17_18-25-09')
    """
    m = BASE_RE.match(stem)
    if m:
        return m.group(1)
    # Fallback: take everything up to last 3 underscores (class, ts, idx)
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-3])
    return stem

# --- Scan folder, build pairs -----------------------------------

pairs_by_base = defaultdict(list)
npys = list(ROOT.rglob("*.npy"))
for npy in npys:
    stem = npy.stem
    rgb = None
    for ext in (".png", ".jpg", ".jpeg"):
        cand = npy.with_suffix(ext)
        if cand.exists():
            rgb = cand
            break
    if rgb is None:
        continue
    bid = base_id_of(stem)
    pairs_by_base[bid].append((rgb, npy))

if not pairs_by_base:
    print("No .npy/.png|.jpg pairs found.")
    raise SystemExit

# --- Compute global per-image min/max for required bands --------

globals_per_base = {}

for bid, pairs in pairs_by_base.items():
    gmin = {}
    gmax = {}
    # First pass: collect mins/maxs across all ROIs for the 3 bands
    for _, npy in pairs:
        cube = read_cube(npy)
        H, W, C = cube.shape
        for idx in (B_IDX, G_IDX, R_IDX):
            idx = clamp_band_idx(idx, C)
            band = cube[..., idx].astype(np.float32)
            mn = float(np.min(band))
            mx = float(np.max(band))
            gmin[idx] = mn if idx not in gmin else min(gmin[idx], mn)
            gmax[idx] = mx if idx not in gmax else max(gmax[idx], mx)
    globals_per_base[bid] = (gmin, gmax)

# --- Compare using global stats & robust channel order ----------

total_pairs = 0
ok_pairs = 0
worst = ("", -1.0)
best = ("", 1e9)

for bid, pairs in pairs_by_base.items():
    gmin, gmax = globals_per_base[bid]
    print(f"\n=== Base: {bid}  (pairs: {len(pairs)}) ===")
    for rgb_path, npy_path in pairs:
        total_pairs += 1
        try:
            img = read_rgb(rgb_path)          # RGB
            cube = read_cube(npy_path)
        except Exception as e:
            print(f"❌ Load error: {rgb_path.name} / {npy_path.name} -> {e}")
            c
