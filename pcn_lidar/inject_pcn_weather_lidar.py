#!/usr/bin/env python3
"""PCN-LiDAR: project weather noise from a generated scan onto clear LiDAR (constant N).

Pipeline
--------
Given:
  - Original scan ``Po`` (.bin): clean geometry
  - Generated/weathered scan ``Pg`` (.bin): weather artifacts (e.g. rain outliers)
Produce:
  - Output ``Pout`` (.bin) with the **same point count** as ``Po``, where weather
    noise is injected in a controlled, severity-aware manner.

Steps (high level)
------------------
1. Robust registration (SE(3) + optional axis-convention search) via multi-stage
   trimmed ICP.
2. Structure / noise separation: residual ``d(x) = min_y ||x - y||`` against ``Po``;
   points with ``d(x) > tau(r) = tau0 + k*r`` are treated as weather outliers.
3. Constant-N injection: replace a fraction of ``Po`` points with mined outliers,
   plus anisotropic jitter, range-dependent intensity attenuation, and dropout
   (near-zero intensity). Point count remains exactly ``N``.

Input format
------------
KITTI-style float32 ``[x, y, z, intensity]`` per point (``DIMS=4``).

Upstream generation typically comes from WeatherGen
(https://github.com/wuyang98/weathergen); this module does not vendor it.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

import numpy as np

# ----------------------------
# Fast NN (SciPy cKDTree preferred)
# ----------------------------
try:
    from scipy.spatial import cKDTree

    def build_tree(X: np.ndarray):
        return cKDTree(X)

    def nn_query(tree, Q: np.ndarray):
        d, idx = tree.query(Q, k=1, workers=-1)
        return d, idx

    HAS_CKDTREE = True
except Exception:  # pragma: no cover - sklearn fallback
    from sklearn.neighbors import KDTree

    def build_tree(X: np.ndarray):
        return KDTree(X, leaf_size=40)

    def nn_query(tree, Q: np.ndarray):
        d, idx = tree.query(Q, k=1)
        return d[:, 0], idx[:, 0]

    HAS_CKDTREE = False

# ----------------------------
# I/O
# ----------------------------
DIMS = 4  # xyz + intensity


def load_bin(path: str, dims: int = DIMS) -> np.ndarray:
    """Load a KITTI-style float32 point cloud from ``path``."""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % dims != 0:
        raise ValueError(f"{path}: size {arr.size} not divisible by dims={dims}")
    return arr.reshape(-1, dims)


def save_bin(path: str, pts: np.ndarray) -> None:
    """Write ``pts`` as float32 binary (KITTI layout)."""
    pts.astype(np.float32).tofile(path)


# ----------------------------
# Basic math utils
# ----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def apply_T(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform ``R, t`` to points ``X`` (Nx3)."""
    return (X @ R.T) + t


def kabsch_se3(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid fit for paired points: minimize ``||R X + t - Y||^2``."""
    if X.shape[0] < 3:
        raise ValueError("Need >=3 correspondences for SE(3).")
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = my - R @ mx
    return R, t

# ----------------------------
# Downsampling & filtering
# ----------------------------
def voxel_downsample_fast(X: np.ndarray, voxel: float) -> np.ndarray:
    """Fast voxel downsample by hashing voxel coords."""
    if X.shape[0] == 0:
        return X
    q = np.floor(X / voxel).astype(np.int32)
    h = (q[:, 0].astype(np.int64) << 42) ^ (q[:, 1].astype(np.int64) << 21) ^ q[:, 2].astype(np.int64)
    _, idx = np.unique(h, return_index=True)
    return X[idx]

def cap_points(X: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    if cap <= 0 or X.shape[0] <= cap:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx]

def filter_structure(X: np.ndarray, r_min=2.0, r_max=80.0, z_min=-2.5, z_max=3.0) -> np.ndarray:
    """Keep a stable structural subset for registration (reduce dominance of ground-noise)."""
    r = np.linalg.norm(X, axis=1)
    m = (r >= r_min) & (r <= r_max) & (X[:, 2] >= z_min) & (X[:, 2] <= z_max)
    return X[m]

# ----------------------------
# Axis convention hypotheses (optional)
# ----------------------------
def generate_B_candidates(include_reflections: bool = True):
    """
    Returns list of 3x3 orthonormal matrices with entries in {-1,0,1}:
      - 24 proper rotations (det=+1)
      - optionally plus 24 reflections (det=-1) => 48 total
    """
    import itertools
    Bs = []
    for perm in itertools.permutations([0, 1, 2]):
        P = np.zeros((3, 3), dtype=np.float64)
        for r, c in enumerate(perm):
            P[r, c] = 1.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    B = np.diag([sx, sy, sz]) @ P
                    det = int(round(np.linalg.det(B)))
                    if det == 1 or (include_reflections and det == -1):
                        Bs.append(B)
    uniq, seen = [], set()
    for B in Bs:
        key = tuple(B.astype(np.int8).flatten().tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(B)
    return uniq

def cheap_score_B(Xg: np.ndarray, Xo: np.ndarray, B: np.ndarray) -> float:
    """Cheap pruning: compare z-hist + range-hist after applying B. Smaller is better."""
    XgB = Xg @ B.T
    zg, zo = XgB[:, 2], Xo[:, 2]
    bins_z = np.linspace(-3, 3, 25)
    hg, _ = np.histogram(zg, bins=bins_z, density=True)
    ho, _ = np.histogram(zo, bins=bins_z, density=True)
    dz = float(np.sum(np.abs(hg - ho)))

    rg = np.linalg.norm(XgB, axis=1)
    ro = np.linalg.norm(Xo, axis=1)
    bins_r = np.linspace(0, 80, 33)
    hgr, _ = np.histogram(rg, bins=bins_r, density=True)
    hor, _ = np.histogram(ro, bins=bins_r, density=True)
    dr = float(np.sum(np.abs(hgr - hor)))

    return dz + 0.3 * dr

# ----------------------------
# Robust trimmed ICP (point-to-point)
# ----------------------------
def trimmed_icp(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    R0=None,
    t0=None,
    iters=15,
    gate=1.0,
    trim=0.8,
    min_inliers=200,
):
    """
    Robust ICP:
      - NN correspondences
      - hard gate d<gate
      - trimming keep best trim fraction (robust to outliers)
      - Kabsch solve
    """
    R = np.eye(3) if R0 is None else R0.copy()
    t = np.zeros(3) if t0 is None else t0.copy()
    tree = build_tree(X_tgt)

    prev_rmse = np.inf
    for _ in range(iters):
        Xw = apply_T(X_src, R, t)
        dists, idx = nn_query(tree, Xw)
        inl = dists < gate
        n_inl = int(inl.sum())
        if n_inl < min_inliers:
            break

        di = dists[inl]
        Xi = X_src[inl]
        Yi = X_tgt[idx[inl]]

        keep = int(max(50, math.floor(trim * di.shape[0])))
        keep = min(keep, di.shape[0])
        sel = np.argpartition(di, keep - 1)[:keep]
        Xi, Yi = Xi[sel], Yi[sel]

        Rn, tn = kabsch_se3(Xi, Yi)
        R, t = Rn, tn

        Xw2 = apply_T(Xi, R, t)
        rmse = float(np.sqrt(np.mean(np.sum((Xw2 - Yi) ** 2, axis=1))))
        if abs(prev_rmse - rmse) < 1e-4:
            break
        prev_rmse = rmse

    Xw = apply_T(X_src, R, t)
    dists, _ = nn_query(tree, Xw)
    fitness = float(np.mean(dists < gate))
    rmse = float(np.sqrt(np.mean(dists[dists < gate] ** 2))) if np.any(dists < gate) else float("inf")
    return R, t, fitness, rmse

def multistage_alignment(Xg: np.ndarray, Xo: np.ndarray, t0: np.ndarray, stages):
    """Coarse-to-fine trimmed ICP."""
    R = np.eye(3)
    t = t0.copy()
    for si, st in enumerate(stages):
        Xo_s = cap_points(voxel_downsample_fast(Xo, st["voxel"]), st["cap"], seed=10 + si)
        Xg_s = cap_points(voxel_downsample_fast(Xg, st["voxel"]), st["cap"], seed=20 + si)
        R, t, fit, rmse = trimmed_icp(
            X_src=Xg_s, X_tgt=Xo_s,
            R0=R, t0=t,
            iters=st["iters"],
            gate=st["gate"],
            trim=st["trim"],
            min_inliers=st["min_inliers"],
        )
    return R, t, fit, rmse

def align_generated_to_original(
    Xg_align: np.ndarray,
    Xo_align: np.ndarray,
    assume_same_coords: bool,
    include_reflections: bool,
    topk: int,
    stages,
    skip_align: bool = False,
):
    """Estimate axis mapping B (optional) and rigid transform (R,t)."""
    if skip_align:
        B = np.eye(3)
        R = np.eye(3)
        t = np.zeros(3, dtype=np.float64)
        return B, R, t, 1.0, 0.0
    if assume_same_coords:
        B = np.eye(3)
        XgB = Xg_align
        t0 = Xo_align.mean(0) - XgB.mean(0)
        R, t, fit, rmse = multistage_alignment(XgB, Xo_align, t0=t0, stages=stages)
        return B, R, t, fit, rmse

    Bs = generate_B_candidates(include_reflections=include_reflections)

    # prune candidates cheaply
    Xo0 = cap_points(voxel_downsample_fast(Xo_align, stages[0]["voxel"]), stages[0]["cap"], seed=1)
    Xg0 = cap_points(voxel_downsample_fast(Xg_align, stages[0]["voxel"]), stages[0]["cap"], seed=2)

    scored = [(cheap_score_B(Xg0, Xo0, B), i) for i, B in enumerate(Bs)]
    scored.sort(key=lambda x: x[0])
    cand = [Bs[i] for _, i in scored[:max(1, topk)]]

    best = None
    for B in cand:
        XgB = Xg_align @ B.T
        t0 = Xo_align.mean(0) - XgB.mean(0)
        R, t, fit, rmse = multistage_alignment(XgB, Xo_align, t0=t0, stages=stages)
        score = (fit, -rmse)
        if best is None or score > best["score"]:
            best = {"B": B, "R": R, "t": t, "fit": fit, "rmse": rmse, "score": score}

    return best["B"], best["R"], best["t"], best["fit"], best["rmse"]

# ----------------------------
# Noise extraction (surface-consistency residual)
# ----------------------------
def noise_mask_from_residual(Xg_aligned: np.ndarray, Xo: np.ndarray, tau0: float, k: float, max_r: float):
    """
    Residual: d(x)=min_{y in Xo} ||x-y||
    Noise if d(x) > tau(r)=tau0 + k*r (range-aware).
    """
    tree = build_tree(Xo)
    d, _ = nn_query(tree, Xg_aligned)
    r = np.linalg.norm(Xg_aligned, axis=1)
    r = np.clip(r, 0.0, max_r)
    tau = tau0 + k * r
    return d > tau, d, r

# ----------------------------
# Intensity mapping (keep original scale)
# ----------------------------
def range_bins(r: np.ndarray, edges: np.ndarray):
    return np.digitize(r, edges) - 1

def build_intensity_sampler(Xo: np.ndarray, Io: np.ndarray, bin_edges: np.ndarray, seed: int = 0):
    """
    Sample intensities from the original distribution conditioned on range bin.
    This preserves original sensor intensity scale statistics.
    """
    r = np.linalg.norm(Xo, axis=1)
    b = range_bins(r, bin_edges)
    buckets = []
    for bi in range(len(bin_edges) - 1):
        vals = Io[b == bi]
        buckets.append(vals if vals.size else None)
    all_vals = Io
    rng = np.random.default_rng(seed)

    def sample_intensity_for_ranges(rq: np.ndarray):
        qb = range_bins(rq, bin_edges)
        out = np.empty_like(rq, dtype=np.float32)
        for i, bi in enumerate(qb):
            vals = buckets[bi] if (0 <= bi < len(buckets)) else None
            if vals is None:
                vals = all_vals
            out[i] = float(vals[rng.integers(0, vals.size)])
        return out

    return sample_intensity_for_ranges

# ----------------------------
# Weather / severity schedules
# ----------------------------
def severity_scalar(severity_name: str, level: Optional[float] = None) -> float:
    """Map named severity (or continuous ``level``) to a scalar in ``[0, 1]``."""
    if level is not None:
        return clamp01(level)
    return {"light": 0.25, "medium": 0.55, "heavy": 0.85}[severity_name]


def params_from_weather(
    weather: str,
    s: float,
    default_replace_frac: Optional[float] = None,
) -> dict[str, float]:
    """Return weather-aware injection parameters for severity scalar ``s``.

    Point count stays constant. Higher severity means a larger ``replace_frac``,
    more jitter / attenuation / dropout, and a more permissive residual threshold
    so enough noise points can be mined from ``Pg``.
    """
    weather = weather.lower()

    if weather == "rain":
        replace_frac = 0.005 + 0.08 * s      # 0.5% .. 8.5%
        tau0 = 0.28 - 0.12 * s               # 0.28 .. 0.16 (more permissive at high s)
        k = 0.003 + 0.004 * s                # 0.003 .. 0.007
        jitter_xy = 0.01 + 0.06 * s          # 1cm .. 7cm
        jitter_z = 0.02 + 0.12 * s           # 2cm .. 14cm
        base_atten = 1.0 - 0.45 * s          # 1.0 .. 0.55
        base_dropout = 0.00 + 0.12 * s       # 0% .. 12%

    elif weather == "fog":
        # fog: fewer spurious points, more attenuation/dropout
        replace_frac = 0.002 + 0.02 * s      # 0.2% .. 2.2%
        tau0 = 0.30 - 0.08 * s
        k = 0.003 + 0.003 * s
        jitter_xy = 0.005 + 0.02 * s
        jitter_z = 0.005 + 0.02 * s
        base_atten = 1.0 - 0.65 * s          # stronger attenuation
        base_dropout = 0.02 + 0.18 * s       # more missing returns

    elif weather == "snow":
        # snow: more spurious points than fog, moderate attenuation
        replace_frac = 0.004 + 0.06 * s
        tau0 = 0.28 - 0.10 * s
        k = 0.003 + 0.004 * s
        jitter_xy = 0.01 + 0.05 * s
        jitter_z = 0.02 + 0.10 * s
        base_atten = 1.0 - 0.50 * s
        base_dropout = 0.01 + 0.14 * s

    else:  # clear
        replace_frac = 0.0
        tau0, k = 0.30, 0.003
        jitter_xy, jitter_z = 0.0, 0.0
        base_atten, base_dropout = 1.0, 0.0

    # default reasonable noise frac if user didn't override:
    if default_replace_frac is not None and default_replace_frac >= 0:
        replace_frac = float(default_replace_frac)

    return dict(
        replace_frac=float(replace_frac),
        tau0=float(tau0),
        k=float(k),
        jitter_xy=float(jitter_xy),
        jitter_z=float(jitter_z),
        base_atten=float(base_atten),
        base_dropout=float(base_dropout),
    )

# ----------------------------
# Constant-N injection with weather control
# ----------------------------
def choose_replacement_indices(Po_xyz: np.ndarray, m: int, prefer_far=True, seed=0):
    rng = np.random.default_rng(seed)
    N = Po_xyz.shape[0]
    if m <= 0:
        return np.array([], dtype=np.int64)
    if prefer_far:
        r = np.linalg.norm(Po_xyz, axis=1)
        thresh = np.quantile(r, 0.70)  # farthest 30%
        candidates = np.where(r >= thresh)[0]
        if candidates.size < m:
            candidates = np.arange(N)
    else:
        candidates = np.arange(N)
    return rng.choice(candidates, size=m, replace=False)

def inject_noise_constantN_weather(
    Po: np.ndarray,
    Xnoise: np.ndarray,
    Inoise: np.ndarray,
    replace_frac: float,
    jitter_xy: float,
    jitter_z: float,
    base_atten: float,
    base_dropout: float,
    range_norm_m: float,
    prefer_far: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Inject weather noise into ``Po`` while keeping the same point count ``N``.

    Replaces ``m`` points with mined noise (plus anisotropic jitter and
    range-dependent attenuation/dropout). Dropout sets intensity near zero but
    keeps the point so ``N`` is unchanged.
    """
    rng = np.random.default_rng(seed)
    out = Po.copy()
    N = out.shape[0]
    m = int(round(replace_frac * N))
    if m <= 0 or Xnoise.shape[0] == 0:
        return out

    m = min(m, Xnoise.shape[0])
    repl_idx = choose_replacement_indices(out[:, :3], m, prefer_far=prefer_far, seed=seed)
    noise_idx = rng.choice(np.arange(Xnoise.shape[0]), size=m, replace=False)

    Xsel = Xnoise[noise_idx].copy()
    Isel = Inoise[noise_idx].copy()

    # anisotropic jitter
    if jitter_xy > 0 or jitter_z > 0:
        jit = np.zeros_like(Xsel, dtype=np.float32)
        jit[:, 0:2] = rng.normal(scale=jitter_xy, size=(m, 2)).astype(np.float32)
        jit[:, 2] = rng.normal(scale=jitter_z, size=(m,)).astype(np.float32)
        Xsel += jit

    # range dependent attenuation/dropout
    r = np.linalg.norm(Xsel, axis=1)
    r_norm = np.clip(r / float(range_norm_m), 0.0, 1.0)

    # attenuation increases with range
    atten = base_atten * (1.0 - 0.25 * r_norm)   # extra up to 25% far-range loss
    Isel = np.clip(Isel * atten, 0.0, 1.0).astype(np.float32)

    # dropout increases with range (simulate more missed returns far away)
    p_drop = np.clip(base_dropout * (0.4 + 0.6 * r_norm), 0.0, 0.9)
    drop = rng.random(m) < p_drop
    Isel[drop] = 0.0  # keep point, but "no return" intensity

    out[repl_idx, :3] = Xsel
    out[repl_idx, 3] = Isel
    return out.astype(np.float32)

# ----------------------------
# Main
# ----------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry: project weather noise from ``--gen`` onto ``--orig``."""
    ap = argparse.ArgumentParser(
        description="PCN-LiDAR: constant-N weather noise projection onto clear scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--orig", required=True, help="Clear/original KITTI .bin (x,y,z,intensity float32).")
    ap.add_argument("--gen", required=True, help="Generated/weathered .bin to mine noise from (e.g. WeatherGen).")
    ap.add_argument("--out", required=True, help="Output .bin path (constant N matching --orig).")

    ap.add_argument(
        "--weather",
        choices=["clear", "rain", "fog", "snow"],
        default="rain",
        help="Weather schedule controlling replace/jitter/attenuation defaults.",
    )
    ap.add_argument(
        "--severity",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Named severity tier (ignored if --level is set).",
    )
    ap.add_argument(
        "--level",
        type=float,
        default=None,
        help="Continuous severity in [0,1]. Overrides --severity mapping.",
    )
    ap.add_argument(
        "--replace_frac",
        type=float,
        default=None,
        help="Override replacement fraction (e.g. 0.03). If omitted, use weather schedule.",
    )
    ap.add_argument(
        "--prefer_far",
        action="store_true",
        help="Replace far points first (recommended).",
    )

    ap.add_argument(
        "--assume_same_coords",
        action="store_true",
        help="Skip axis-convention search; still run ICP translation/rotation.",
    )
    ap.add_argument(
        "--skip_align",
        action="store_true",
        help="Ablation: identity transform (no ICP / no axis search).",
    )
    ap.add_argument(
        "--include_reflections",
        action="store_true",
        help="Include det=-1 axis candidates when searching conventions.",
    )
    ap.add_argument("--topk", type=int, default=6, help="Top-K axis candidates after cheap pruning.")

    # Explicit schedule overrides (None = keep weather/severity defaults)
    ap.add_argument("--tau0", type=float, default=None, help="Override residual tau0.")
    ap.add_argument("--k", type=float, default=None, help="Override residual slope k.")
    ap.add_argument("--sigma_xy", type=float, default=None, help="Override jitter_xy (m).")
    ap.add_argument("--sigma_z", type=float, default=None, help="Override jitter_z (m).")
    ap.add_argument("--base_atten", type=float, default=None, help="Override base intensity attenuation.")
    ap.add_argument("--base_dropout", type=float, default=None, help="Override base dropout probability.")
    ap.add_argument("--intensity_source", choices=["po_bins", "pg"], default="po_bins",
                    help="po_bins: range-binned resample from Po; pg: use Pg intensities on mined points.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for intensity sampling + injection.")

    # alignment subset filter
    ap.add_argument("--rmin", type=float, default=2.0)
    ap.add_argument("--rmax", type=float, default=80.0)
    ap.add_argument("--zmin", type=float, default=-2.5)
    ap.add_argument("--zmax", type=float, default=3.0)

    # residual threshold cap & normalization range for weather effects
    ap.add_argument("--max_r", type=float, default=120.0, help="Cap r when computing tau(r).")
    ap.add_argument("--range_norm", type=float, default=80.0, help="Range (m) used to normalize attenuation/dropout effects.")

    # intensity sampling bins
    ap.add_argument("--bin_edges", type=str, default="0,10,20,30,40,50,60,70,80,100,120")

    # ICP schedule knobs
    ap.add_argument("--cap", type=int, default=15000)
    ap.add_argument("--v0", type=float, default=0.40)
    ap.add_argument("--v1", type=float, default=0.20)
    ap.add_argument("--v2", type=float, default=0.10)
    ap.add_argument("--g0", type=float, default=2.00)
    ap.add_argument("--g1", type=float, default=1.00)
    ap.add_argument("--g2", type=float, default=0.50)
    ap.add_argument("--t0", type=float, default=0.70)
    ap.add_argument("--t1", type=float, default=0.80)
    ap.add_argument("--t2", type=float, default=0.85)
    ap.add_argument("--i0", type=int, default=10)
    ap.add_argument("--i1", type=int, default=10)
    ap.add_argument("--i2", type=int, default=5)
    ap.add_argument("--min_inliers", type=int, default=200)

    args = ap.parse_args(argv)

    # --- load ---
    Po = load_bin(args.orig, DIMS)
    Pg = load_bin(args.gen, DIMS)
    Xo, Io = Po[:, :3], Po[:, 3]
    Xg = Pg[:, :3]

    # --- severity + weather schedules (default reasonable noise frac comes from schedule) ---
    s = severity_scalar(args.severity, args.level)
    P = params_from_weather(args.weather, s, args.replace_frac)
    if args.tau0 is not None:
        P["tau0"] = float(args.tau0)
    if args.k is not None:
        P["k"] = float(args.k)
    if args.sigma_xy is not None:
        P["jitter_xy"] = float(args.sigma_xy)
    if args.sigma_z is not None:
        P["jitter_z"] = float(args.sigma_z)
    if args.base_atten is not None:
        P["base_atten"] = float(args.base_atten)
    if args.base_dropout is not None:
        P["base_dropout"] = float(args.base_dropout)

    # --- alignment subsets ---
    Xo_align = filter_structure(Xo, r_min=args.rmin, r_max=args.rmax, z_min=args.zmin, z_max=args.zmax)
    Xg_align = filter_structure(Xg, r_min=args.rmin, r_max=args.rmax, z_min=args.zmin, z_max=args.zmax)
    if Xo_align.shape[0] < 500 or Xg_align.shape[0] < 500:
        raise RuntimeError("Too few points after filtering; relax r/z filters or check input files.")

    stages = [
        {"voxel": args.v0, "cap": args.cap, "gate": args.g0, "trim": args.t0, "iters": args.i0, "min_inliers": args.min_inliers},
        {"voxel": args.v1, "cap": args.cap, "gate": args.g1, "trim": args.t1, "iters": args.i1, "min_inliers": args.min_inliers},
        {"voxel": args.v2, "cap": args.cap, "gate": args.g2, "trim": args.t2, "iters": args.i2, "min_inliers": args.min_inliers},
    ]

    # --- robust align ---
    B, R, t, fit, rmse = align_generated_to_original(
        Xg_align=Xg_align,
        Xo_align=Xo_align,
        assume_same_coords=args.assume_same_coords,
        include_reflections=args.include_reflections,
        topk=args.topk,
        stages=stages,
        skip_align=args.skip_align,
    )
    if args.skip_align:
        print("[align] skipped (identity); tree=%s" % ("cKDTree" if HAS_CKDTREE else "sklearnKDTree"))
    else:
        print(f"[align] tree={'cKDTree' if HAS_CKDTREE else 'sklearnKDTree'} fit={fit:.3f} rmse={rmse:.3f}")

    # apply alignment to full generated
    Xg_aligned = apply_T(Xg @ B.T, R, t)
    Ig = Pg[:, 3]

    # --- mine noise candidates by residual (surface-consistency) ---
    noise_mask, residuals, _ranges = noise_mask_from_residual(
        Xg_aligned, Xo, tau0=P["tau0"], k=P["k"], max_r=args.max_r
    )
    Xnoise = Xg_aligned[noise_mask]
    Inoise_pg = Ig[noise_mask]
    print(f"[noise] mined {Xnoise.shape[0]} / {Xg_aligned.shape[0]} noise candidates | weather={args.weather} s={s:.2f}")

    # ensure enough noise pool (need >~ 10x replacement budget)
    target_pool = int(max(500, 10.0 * P["replace_frac"] * Po.shape[0]))
    if Xnoise.shape[0] < target_pool:
        top = min(target_pool, Xg_aligned.shape[0])
        idx = np.argsort(residuals)[-top:]
        Xnoise = Xg_aligned[idx]
        Inoise_pg = Ig[idx]
        print(f"[noise] fallback: using top-{top} residual points as noise pool")

    # --- intensity mapping ---
    if args.intensity_source == "pg":
        Inoise = Inoise_pg.astype(np.float32)
    else:
        edges = np.array([float(x) for x in args.bin_edges.split(",")], dtype=np.float64)
        sampler = build_intensity_sampler(Xo, Io, edges, seed=args.seed)
        Inoise = sampler(np.linalg.norm(Xnoise, axis=1))

    # --- constant-N injection ---
    out = inject_noise_constantN_weather(
        Po, Xnoise, Inoise,
        replace_frac=P["replace_frac"],
        jitter_xy=P["jitter_xy"],
        jitter_z=P["jitter_z"],
        base_atten=P["base_atten"],
        base_dropout=P["base_dropout"],
        range_norm_m=args.range_norm,
        prefer_far=args.prefer_far,
        seed=args.seed,
    )

    save_bin(args.out, out)

    print("[params] "
          f"replace_frac={P['replace_frac']:.4f} tau0={P['tau0']:.3f} k={P['k']:.4f} "
          f"jxy={P['jitter_xy']:.3f} jz={P['jitter_z']:.3f} "
          f"atten={P['base_atten']:.3f} dropout={P['base_dropout']:.3f}")
    print(f"[done] wrote {args.out} with N={out.shape[0]} (same as original N={Po.shape[0]})")

if __name__ == "__main__":
    main()