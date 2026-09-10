#!/usr/bin/env python3
"""Batch: clear LiDAR -> WeatherGen -> PCN projection (light / medium / heavy).

Pipeline per weather (rain / fog / snow):
  1. Convert clear scans to KITTI-4 float32 if needed (Waymo 6-D is truncated).
  2. Run WeatherGen ``generate.py`` once to produce weathered ``generated.bin``.
  3. For each severity, run ``inject_pcn_weather_lidar.py`` (PCN) to inject noise
     onto clear geometry at constant point count.

Outputs (folder naming matches existing rain_* convention)::

  <split_root>/{weather}_light|moderate|severe/{lidar_subdir}/

PCN severities map as: light / medium / heavy
(folder suffixes light / moderate / severe).

KITTI validation (flat clear velodyne, no clear/ parent)::

  --kitti-clear-lidar .../validation/velodyne
  --kitti-out-root    .../validation
  writes velodyne_{weather}_{light|moderate|severe}/*.bin

Intermediate cache: ``<cache_root>/_weathergen_pcn_cache/<dataset_tag>/<weather>/``

WeatherGen is an external dependency — set ``--weathergen-root`` or env
``WEATHERGEN_ROOT`` (default: sibling ``../weathergen`` if present).
Official repo: https://github.com/wuyang98/weathergen
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

WAYMO_DIMS = 6
KITTI_DIMS = 4

# (output folder suffix, PCN --severity)
SEVERITY_DIRS = (
    ("light", "light"),
    ("moderate", "medium"),
    ("severe", "heavy"),
)

WEATHERS = ("rain", "fog", "snow")

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_INJECT_PCN = _PKG_DIR / "inject_pcn_weather_lidar.py"


def _default_weathergen_root() -> Path:
    env = os.environ.get("WEATHERGEN_ROOT", "").strip()
    if env:
        return Path(env)
    # Common local layout next to this toolkit / under Test_Code
    candidates = [
        _PKG_DIR.parent.parent / "weathergen",
        Path("/home/wael/Test_Code/weathergen"),
    ]
    for c in candidates:
        if (c / "generate.py").is_file():
            return c
    return candidates[0]


def load_bin_any(path: Path) -> tuple[np.ndarray, int]:
    """Load float32 .bin as Waymo-6 or KITTI-4; return ``(points, dims)``."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % WAYMO_DIMS == 0:
        return raw.reshape(-1, WAYMO_DIMS), WAYMO_DIMS
    if raw.size % KITTI_DIMS == 0:
        return raw.reshape(-1, KITTI_DIMS), KITTI_DIMS
    raise ValueError(f"{path}: cannot reshape float32 as 6- or 4-dim ({raw.size} floats)")


def to_kitti4(points: np.ndarray, dims: int) -> np.ndarray:
    """Truncate Waymo-6 to KITTI-4 or pass through KITTI-4."""
    if dims == KITTI_DIMS:
        return points.astype(np.float32, copy=False)
    if dims == WAYMO_DIMS:
        return points[:, :KITTI_DIMS].astype(np.float32, copy=False)
    raise ValueError(dims)


def merge_waymo(pcn_out4: np.ndarray, po6: np.ndarray) -> np.ndarray:
    """Restore Waymo extra channels after PCN (which operates in 4-D)."""
    assert pcn_out4.shape[0] == po6.shape[0]
    return np.column_stack(
        [pcn_out4[:, 0], pcn_out4[:, 1], pcn_out4[:, 2], pcn_out4[:, 3], po6[:, 4], po6[:, 5]]
    ).astype(np.float32)


def ensure_symlink_dir(link_path: Path, target_relative: Path) -> None:
    """Create or refresh a directory symlink; no-op if a real dir already exists."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        cur = os.readlink(link_path)
        if cur != str(target_relative):
            link_path.unlink()
            link_path.symlink_to(target_relative, target_is_directory=True)
        return
    if link_path.exists():
        return
    link_path.symlink_to(target_relative, target_is_directory=True)


def save_bin(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts.astype(np.float32).tofile(path)


def run_weathergen_batch(
    *,
    python_exe: str,
    weathergen_root: Path,
    generate_py: Path,
    kitti4_dir: Path,
    wg_out: Path,
    weather: str,
    ckpt: Path,
    device: str,
    batch_size: int,
    sampling_steps: int,
    max_samples: Optional[int],
) -> None:
    """Invoke WeatherGen ``generate.py`` on a directory of KITTI-4 bins."""
    wg_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(generate_py),
        "--input_dir",
        str(kitti4_dir),
        "--weather",
        weather,
        "--output_dir",
        str(wg_out),
        "--ckpt",
        str(ckpt),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--sampling_steps",
        str(sampling_steps),
    ]
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    print("[WeatherGen]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(weathergen_root), check=True)


def run_pcn(
    *,
    python_exe: str,
    inject_pcn: Path,
    orig4: Path,
    gen_bin: Path,
    out_path: Path,
    weather: str,
    severity: str,
    prefer_far: bool,
) -> None:
    """Run PCN once; on failure, retry with relaxed structure filters."""
    cmd = [
        python_exe,
        str(inject_pcn),
        "--orig",
        str(orig4),
        "--gen",
        str(gen_bin),
        "--out",
        str(out_path),
        "--weather",
        weather,
        "--severity",
        severity,
    ]
    if prefer_far:
        cmd.append("--prefer_far")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        relaxed_cmd = cmd + [
            "--rmin",
            "0.0",
            "--rmax",
            "120.0",
            "--zmin",
            "-5.0",
            "--zmax",
            "5.0",
            "--min_inliers",
            "50",
        ]
        print("[PCN retry relaxed]", " ".join(relaxed_cmd), flush=True)
        subprocess.run(relaxed_cmd, check=True)


def process_clear_lidar(
    *,
    clear_lidar: Path,
    out_root: Path,
    dataset_tag: str,
    weather: str,
    lidar_subdir: Optional[str],
    python_exe: str,
    weathergen_root: Path,
    generate_py: Path,
    inject_pcn: Path,
    ckpt: Path,
    device: str,
    batch_size: int,
    sampling_steps: int,
    skip_weathergen: bool,
    skip_pcn: bool,
    prefer_far: bool,
    limit: Optional[int],
    kitti_flat_out: bool,
    symlink_images: bool,
) -> None:
    """Process one clear-lidar directory for a single weather condition."""
    clear_lidar = clear_lidar.resolve()
    out_root = out_root.resolve()
    if not clear_lidar.is_dir():
        raise FileNotFoundError(clear_lidar)

    cache = out_root / "_weathergen_pcn_cache" / dataset_tag / weather
    kitti4_dir = cache / "kitti4_input"
    wg_out = cache / "weathergen_out"
    manifest_path = cache / "manifest.json"

    bins = sorted(clear_lidar.glob("*.bin"))
    if limit is not None:
        bins = bins[: max(0, limit)]
    if not bins:
        raise RuntimeError(f"No .bin under {clear_lidar}")

    kitti4_dir.mkdir(parents=True, exist_ok=True)

    stems: list[str] = []
    originals: dict[str, tuple[np.ndarray, int]] = {}

    for p in bins:
        pts, dims = load_bin_any(p)
        originals[p.name] = (pts, dims)
        stems.append(p.name)
        k4_path = kitti4_dir / p.name
        if k4_path.exists():
            continue
        k4 = to_kitti4(pts, dims)
        save_bin(k4_path, k4)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stems": stems,
                "weather": weather,
                "clear_lidar": str(clear_lidar),
                "lidar_subdir": lidar_subdir,
                "kitti_flat_out": kitti_flat_out,
            },
            f,
            indent=2,
        )

    gen_dirs = {s: wg_out / weather / Path(s).stem for s in stems}

    if not skip_weathergen:
        need_wg = not all((d / "generated.bin").exists() for d in gen_dirs.values())
        if need_wg:
            run_weathergen_batch(
                python_exe=python_exe,
                weathergen_root=weathergen_root,
                generate_py=generate_py,
                kitti4_dir=kitti4_dir,
                wg_out=wg_out,
                weather=weather,
                ckpt=ckpt,
                device=device,
                batch_size=batch_size,
                sampling_steps=sampling_steps,
                max_samples=limit,
            )

    if skip_pcn:
        return

    clear_img_rel = Path("..") / "clear" / "image_0"

    for folder_suffix, _sev in SEVERITY_DIRS:
        if kitti_flat_out:
            out_lidar = out_root / f"velodyne_{weather}_{folder_suffix}"
            out_lidar.mkdir(parents=True, exist_ok=True)
        else:
            folder_name = f"{weather}_{folder_suffix}"
            out_lidar = out_root / folder_name / (lidar_subdir or "lidar")
            out_lidar.mkdir(parents=True, exist_ok=True)
            if symlink_images:
                ensure_symlink_dir(out_root / folder_name / "image_0", clear_img_rel)

    for stem in stems:
        pts, dims = originals[stem]
        k4_path = kitti4_dir / stem
        gen_bin = gen_dirs[stem] / "generated.bin"
        if not gen_bin.is_file():
            raise FileNotFoundError(f"Missing WeatherGen output: {gen_bin}")

        for folder_suffix, severity in SEVERITY_DIRS:
            if kitti_flat_out:
                out_bin = out_root / f"velodyne_{weather}_{folder_suffix}" / stem
            else:
                folder_name = f"{weather}_{folder_suffix}"
                out_bin = out_root / folder_name / (lidar_subdir or "lidar") / stem
            if out_bin.exists():
                continue
            tmp4 = cache / "pcn_tmp" / f"{weather}_{folder_suffix}" / stem
            tmp4.parent.mkdir(parents=True, exist_ok=True)
            run_pcn(
                python_exe=python_exe,
                inject_pcn=inject_pcn,
                orig4=k4_path,
                gen_bin=gen_bin,
                out_path=tmp4,
                weather=weather,
                severity=severity,
                prefer_far=prefer_far,
            )
            pcn = np.fromfile(tmp4, dtype=np.float32).reshape(-1, KITTI_DIMS)
            if dims == WAYMO_DIMS:
                merged = merge_waymo(pcn, pts)
                merged.astype(np.float32).tofile(out_bin)
            else:
                pcn.astype(np.float32).tofile(out_bin)
            print(f"[done] {weather}_{folder_suffix} {stem}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry for WeatherGen + PCN batch generation."""
    wg_root_default = _default_weathergen_root()
    ap = argparse.ArgumentParser(
        description="Clear LiDAR → WeatherGen → PCN (light/medium/heavy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--waymo-root", type=Path, default=None, help="Waymo split root with clear/velodyne.")
    ap.add_argument("--nuscenes-root", type=Path, default=None, help="nuScenes split root with clear/lidar.")
    ap.add_argument(
        "--kitti-clear-lidar",
        type=Path,
        default=None,
        help="KITTI clear velodyne dir (e.g. validation/velodyne).",
    )
    ap.add_argument(
        "--kitti-out-root",
        type=Path,
        default=None,
        help="Where to write velodyne_{weather}_{sev}/ (default: parent of clear lidar).",
    )
    ap.add_argument(
        "--weather",
        type=str,
        default="rain",
        choices=WEATHERS,
        help="Single weather to generate.",
    )
    ap.add_argument(
        "--weathers",
        type=str,
        default=None,
        help="Comma-separated weathers (overrides --weather), e.g. fog,snow.",
    )
    ap.add_argument("--python", type=str, default=sys.executable, help="Python for WeatherGen / PCN.")
    ap.add_argument(
        "--weathergen-root",
        type=Path,
        default=wg_root_default,
        help="Path to WeatherGen checkout (or set WEATHERGEN_ROOT).",
    )
    ap.add_argument(
        "--inject-pcn",
        "--project-noise",
        dest="inject_pcn",
        type=Path,
        default=_DEFAULT_INJECT_PCN,
        help="Path to inject_pcn_weather_lidar.py (PCN core).",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="WeatherGen checkpoint (default: <weathergen-root>/checkpoint/diffusion_0000100000.pth).",
    )
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for WeatherGen.")
    ap.add_argument("--batch-size", type=int, default=4, help="WeatherGen batch size.")
    ap.add_argument("--sampling-steps", type=int, default=256, help="WeatherGen diffusion sampling steps.")
    ap.add_argument("--skip-weathergen", action="store_true", help="Reuse cached WeatherGen outputs.")
    ap.add_argument("--skip-pcn", action="store_true", help="Only run WeatherGen (or skip if cached).")
    ap.add_argument("--no-prefer-far", action="store_true", help="Disable prefer-far replacement in PCN.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N frames (smoke).")
    ap.add_argument("--no-symlink-images", action="store_true", help="Do not symlink clear/image_0.")
    args = ap.parse_args(argv)

    weathergen_root = args.weathergen_root.resolve()
    generate_py = weathergen_root / "generate.py"
    ckpt = (
        args.ckpt.resolve()
        if args.ckpt
        else (weathergen_root / "checkpoint" / "diffusion_0000100000.pth")
    )
    if not args.skip_weathergen and not generate_py.is_file():
        ap.error(
            f"WeatherGen generate.py not found at {generate_py}. "
            "Clone https://github.com/wuyang98/weathergen and pass --weathergen-root."
        )

    prefer_far = not args.no_prefer_far
    weathers = (
        [w.strip() for w in args.weathers.split(",") if w.strip()]
        if args.weathers
        else [args.weather]
    )
    for w in weathers:
        if w not in WEATHERS:
            ap.error(f"Unknown weather {w}; choose from {WEATHERS}")

    common = dict(
        python_exe=args.python,
        weathergen_root=weathergen_root,
        generate_py=generate_py,
        inject_pcn=args.inject_pcn.resolve(),
        ckpt=ckpt,
        device=args.device,
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        skip_weathergen=args.skip_weathergen,
        skip_pcn=args.skip_pcn,
        prefer_far=prefer_far,
        limit=args.limit,
    )

    ran = False
    for weather in weathers:
        if args.waymo_root:
            ran = True
            process_clear_lidar(
                clear_lidar=args.waymo_root.resolve() / "clear" / "velodyne",
                out_root=args.waymo_root.resolve(),
                dataset_tag="waymo",
                weather=weather,
                lidar_subdir="velodyne",
                kitti_flat_out=False,
                symlink_images=not args.no_symlink_images,
                **common,
            )
        if args.nuscenes_root:
            ran = True
            process_clear_lidar(
                clear_lidar=args.nuscenes_root.resolve() / "clear" / "lidar",
                out_root=args.nuscenes_root.resolve(),
                dataset_tag="nuscenes",
                weather=weather,
                lidar_subdir="lidar",
                kitti_flat_out=False,
                symlink_images=not args.no_symlink_images,
                **common,
            )
        if args.kitti_clear_lidar:
            ran = True
            clear = args.kitti_clear_lidar.resolve()
            out_root = args.kitti_out_root.resolve() if args.kitti_out_root else clear.parent
            process_clear_lidar(
                clear_lidar=clear,
                out_root=out_root,
                dataset_tag="kitti",
                weather=weather,
                lidar_subdir=None,
                kitti_flat_out=True,
                symlink_images=False,
                **common,
            )

    if not ran:
        ap.error("Pass --waymo-root and/or --nuscenes-root and/or --kitti-clear-lidar")


if __name__ == "__main__":
    main()
