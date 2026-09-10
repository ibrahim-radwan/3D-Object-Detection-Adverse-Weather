#!/usr/bin/env python3
"""CLI: drive Qwen Image Edit via ComfyUI for one weather/intensity.

Queues API-format workflows against a running ComfyUI server, then lightly
post-processes each PNG so object geometry stays consistent with the clear input.

Example::

    python run_qwen_weather_rgb.py \\
      --weather fog --intensity medium \\
      --input-dir /path/to/clear/image_0 \\
      --output-dir /path/to/output/fog/medium

Requires: requests, numpy, Pillow, tqdm; a reachable ComfyUI with the Qwen
Image Edit workflow nodes installed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

try:
    from .comfyui_http_client import ComfyClient
    from .qwen_weather_prompts import (
        WEATHER_PROMPTS,
        get_negative_prompt,
        get_prompt,
        list_intensities,
        list_weathers,
        load_prompts_json,
    )
except ImportError:  # script execution: ``python run_qwen_weather_rgb.py``
    from comfyui_http_client import ComfyClient
    from qwen_weather_prompts import (
        WEATHER_PROMPTS,
        get_negative_prompt,
        get_prompt,
        list_intensities,
        list_weathers,
        load_prompts_json,
    )

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_WORKFLOW = _PKG_DIR / "workflow_qwen_image_edit.json"
_DEFAULT_PROMPTS_JSON = _PKG_DIR / "qwen_weather_prompts.json"

# Postprocess defaults (geometry-preserving blend)
DIFF_THRESHOLD = 10
MASK_BLUR_RADIUS = 1
PNG_COMPRESS_LEVEL = 1
WEATHER_ALPHA_CAP = {"light": 0.35, "medium": 0.50, "heavy": 0.65}
EDGE_PROTECT_STRENGTH = 0.60
NEAR_FIELD_PROTECT_STRENGTH = 0.35
MAX_COLOR_SHIFT = {"light": 26.0, "medium": 36.0, "heavy": 48.0}

# Throughput / polling
MAX_QUEUE_SIZE = 50
MAX_INFLIGHT = 8
POLL_INTERVAL_SEC = 0.75
HISTORY_TIMEOUT_SEC = 1200


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Build CLI for RGB weather generation."""
    p = argparse.ArgumentParser(
        description="Generate weather-conditioned RGB images with Qwen Image Edit (ComfyUI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="Folder of source images (.jpg / .png).")
    p.add_argument("--output-dir", required=True, help="Destination for generated PNGs.")
    p.add_argument(
        "--weather",
        required=True,
        help=f"Condition key (e.g. fog, rain, night_rain). Known: {', '.join(list_weathers())}",
    )
    p.add_argument(
        "--intensity",
        "--rain-intensity",
        dest="intensity",
        choices=("light", "medium", "heavy"),
        default="medium",
        help="Prompt intensity tier.",
    )
    p.add_argument("--workflow", default=str(_DEFAULT_WORKFLOW), help="ComfyUI API workflow JSON.")
    p.add_argument(
        "--prompts",
        default=None,
        help="Optional qwen_weather_prompts.json override. Default: use built-in "
        f"qwen_weather_prompts.py (or {_DEFAULT_PROMPTS_JSON} if --prompts-json-builtin).",
    )
    p.add_argument(
        "--prompts-json-builtin",
        action="store_true",
        help=f"Load prompts from packaged {_DEFAULT_PROMPTS_JSON.name} instead of "
        "qwen_weather_prompts.py.",
    )
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI HTTP base URL.")
    p.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    p.add_argument("--dry-run", action="store_true", help="Print paths and exit without calling ComfyUI.")
    p.add_argument(
        "--staging-dir",
        default=None,
        metavar="DIR",
        help="Optional: WAS Image Save writes here first, then files move to --output-dir. "
        "Also honored via env COMFY_QWEN_STAGING.",
    )
    return p.parse_args(argv)


@lru_cache(maxsize=512)
def load_original_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.copy()


def safe_open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def difference_mask(orig: Image.Image, gen: Image.Image) -> np.ndarray:
    o = np.asarray(orig, np.int16)
    g = np.asarray(gen, np.int16)
    diff = np.abs(g - o).max(axis=2)
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255
    m = Image.fromarray(mask, "L")
    if MASK_BLUR_RADIUS > 0:
        m = m.filter(ImageFilter.GaussianBlur(MASK_BLUR_RADIUS))
    return np.asarray(m, np.float32) / 255.0


def composite(orig: Image.Image, gen: Image.Image, alpha: np.ndarray) -> Image.Image:
    o = np.asarray(orig, np.float32)
    g = np.asarray(gen, np.float32)
    out = alpha[..., None] * g + (1 - alpha[..., None]) * o
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def semantic_preserve_alpha(
    orig: Image.Image, alpha: np.ndarray, intensity: str
) -> np.ndarray:
    """Suppress weather edits on likely object boundaries / near field."""
    o = np.asarray(orig, np.float32)
    gray = 0.299 * o[:, :, 0] + 0.587 * o[:, :, 1] + 0.114 * o[:, :, 2]
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    g95 = np.percentile(grad, 95.0) + 1e-6
    edge_norm = np.clip(grad / g95, 0.0, 1.0)

    h, _ = gray.shape
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    near_field = np.clip((y - 0.55) / 0.45, 0.0, 1.0)

    alpha_cap = WEATHER_ALPHA_CAP.get(intensity, 0.50)
    preserved = alpha * (1.0 - EDGE_PROTECT_STRENGTH * edge_norm)
    preserved = preserved * (1.0 - NEAR_FIELD_PROTECT_STRENGTH * near_field)
    return np.clip(preserved, 0.0, alpha_cap)


def clamp_color_shift(orig: Image.Image, gen: Image.Image, intensity: str) -> Image.Image:
    """Limit large color/style drift that hurts semantic consistency."""
    o = np.asarray(orig, np.float32)
    g = np.asarray(gen, np.float32)
    lim = MAX_COLOR_SHIFT.get(intensity, 36.0)
    delta = np.clip(g - o, -lim, lim)
    out = np.clip(o + delta, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def save_png(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, "PNG", optimize=False, compress_level=PNG_COMPRESS_LEVEL)


def postprocess(
    original_path: str,
    generated_path: str,
    out_path: str,
    intensity: str = "medium",
) -> None:
    """Blend generated weather onto the original with edge/near-field protection."""
    orig = load_original_rgb(original_path)
    gen = safe_open_rgb(generated_path)
    gen = gen.crop((0, 0, orig.width, orig.height))
    gen = clamp_color_shift(orig, gen, intensity)
    alpha = difference_mask(orig, gen)
    alpha = semantic_preserve_alpha(orig, alpha, intensity)
    final = composite(orig, gen, alpha)
    save_png(final, out_path)


def resolve_prompts(
    weather: str, intensity: str, prompts_path: Optional[str], use_json_builtin: bool
) -> tuple[str, str]:
    """Return ``(positive, negative)`` prompt strings."""
    if prompts_path:
        negative, weather_map = load_prompts_json(prompts_path)
        try:
            return weather_map[weather][intensity], negative
        except KeyError as exc:
            raise SystemExit(
                f"[error] prompts file missing weather_prompts.{weather}.{intensity}"
            ) from exc
    if use_json_builtin:
        negative, weather_map = load_prompts_json(_DEFAULT_PROMPTS_JSON)
        try:
            return weather_map[weather][intensity], negative
        except KeyError as exc:
            raise SystemExit(
                f"[error] qwen_weather_prompts.json missing "
                f"weather_prompts.{weather}.{intensity}"
            ) from exc
    try:
        return get_prompt(weather, intensity), get_negative_prompt()
    except KeyError as exc:
        raise SystemExit(f"[error] {exc}") from exc


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = os.path.realpath(os.path.abspath(args.input_dir))
    out_dir = os.path.realpath(os.path.abspath(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    if args.weather not in WEATHER_PROMPTS and not (args.prompts or args.prompts_json_builtin):
        print(
            f"[error] Unknown weather '{args.weather}'. Known: {list_weathers()}",
            file=sys.stderr,
        )
        return 1
    if (
        args.weather in WEATHER_PROMPTS
        and args.intensity not in list_intensities(args.weather)
        and not (args.prompts or args.prompts_json_builtin)
    ):
        print(
            f"[error] Intensity '{args.intensity}' not defined for '{args.weather}'. "
            f"Available: {list_intensities(args.weather)}",
            file=sys.stderr,
        )
        return 1

    with open(args.workflow, "r", encoding="utf-8") as f:
        base_workflow = json.load(f)

    positive, global_negative = resolve_prompts(
        args.weather, args.intensity, args.prompts, args.prompts_json_builtin
    )

    if not os.path.isdir(input_dir):
        print(f"[error] Input dir not found: {input_dir}", file=sys.stderr)
        return 1

    images = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if args.limit is not None:
        images = images[: max(0, args.limit)]

    if args.dry_run:
        print(f"[dry-run] Would process {len(images)} images")
        print(f"  weather/intensity: {args.weather}/{args.intensity}")
        print(f"  input:  {input_dir}")
        print(f"  output: {out_dir}")
        print(f"  positive prompt (first 120 chars): {positive[:120]!r}")
        return 0

    if not images:
        print(f"[error] No images under {input_dir}", file=sys.stderr)
        return 1

    client = ComfyClient(args.comfy_url)

    if not client.health():
        print(
            f"[error] ComfyUI not reachable at {client.base_url}. Start ComfyUI then re-run.",
            file=sys.stderr,
        )
        return 2

    staging_dir: Optional[str] = args.staging_dir
    if not staging_dir and os.environ.get("COMFY_QWEN_STAGING", "").strip():
        staging_dir = os.environ.get("COMFY_QWEN_STAGING", "").strip()
    if staging_dir:
        staging_dir = os.path.realpath(os.path.abspath(staging_dir))
        os.makedirs(staging_dir, exist_ok=True)
        print(f"[save] staging: Comfy writes to {staging_dir}, then move -> {out_dir}", flush=True)
    else:
        print(f"[save] direct: node 132 output_path = {out_dir}", flush=True)

    write_root = staging_dir if staging_dir else out_dir
    print(f"[paths] Final PNG directory: {out_dir}", flush=True)

    # Workflow node IDs for the packaged Qwen Image Edit API graph.
    workflow_t = copy.deepcopy(base_workflow)
    workflow_t["115:111"]["inputs"]["prompt"] = positive
    workflow_t["115:110"]["inputs"]["prompt"] = global_negative
    workflow_t["134"]["inputs"]["path"] = input_dir
    workflow_t["134"]["inputs"]["index"] = 0
    workflow_t["132"]["inputs"]["output_path"] = write_root
    workflow_t["132"]["inputs"]["overwrite_mode"] = "prefix_as_filename"
    workflow_t["132"]["inputs"]["filename_delimiter"] = "_"

    inflight: dict[str, dict] = {}

    def drain(force: bool = False) -> None:
        while inflight:
            done = [
                pid
                for pid, job in inflight.items()
                if client.is_done(pid) or (time.time() - job["t0"] > HISTORY_TIMEOUT_SEC)
            ]
            if not done:
                if force:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                return

            for pid in done:
                job = inflight.pop(pid)
                gen_path = job["out_path"]
                if staging_dir:
                    staged = os.path.join(staging_dir, os.path.basename(gen_path))
                    if os.path.isfile(staged):
                        try:
                            shutil.move(staged, gen_path)
                        except OSError as exc:
                            print(f"[warn] move {staged} -> {gen_path}: {exc}", flush=True)
                    if not os.path.exists(gen_path) and os.path.isfile(staged):
                        try:
                            shutil.copy2(staged, gen_path)
                        except OSError as exc:
                            print(f"[warn] copy {staged} -> {gen_path}: {exc}", flush=True)
                if os.path.exists(gen_path):
                    postprocess(job["orig"], gen_path, gen_path, intensity=job["intensity"])
                else:
                    timed_out = time.time() - job["t0"] > HISTORY_TIMEOUT_SEC
                    hist_done = client.is_done(pid)
                    if hist_done:
                        print(
                            "[warn] Comfy reports finished outputs but PNG is missing "
                            f"(WAS Image Save). Expected: {gen_path}",
                            flush=True,
                        )
                    elif timed_out:
                        print(
                            "[warn] History timeout before completion; file may still appear later: "
                            f"{gen_path}",
                            flush=True,
                        )

    desc = f"{args.intensity} {args.weather} (Qwen)"
    with tqdm(total=len(images), desc=desc, mininterval=0, miniters=1) as pbar:
        for fname in images:
            name = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{name}.png")
            if os.path.exists(out_path):
                pbar.update(1)
                continue

            while len(inflight) >= MAX_INFLIGHT or client.queue_size() >= MAX_QUEUE_SIZE:
                drain()
                time.sleep(POLL_INTERVAL_SEC)

            wf = copy.deepcopy(workflow_t)
            wf["134"]["inputs"]["pattern"] = fname
            wf["132"]["inputs"]["filename_prefix"] = name

            pid = client.queue_prompt(wf)
            if pid:
                inflight[pid] = {
                    "t0": time.time(),
                    "orig": os.path.join(input_dir, fname),
                    "out_path": out_path,
                    "intensity": args.intensity,
                }
            pbar.update(1)

        drain(force=True)

    png_n = sum(
        1
        for f in os.listdir(out_dir)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(out_dir, f))
    )
    print(f"[done] Output directory: {out_dir} ({png_n} PNG files)", flush=True)
    if png_n == 0:
        extra = (
            f" and staging folder {staging_dir} for leftover PNGs."
            if staging_dir
            else " (Comfy should write directly under --output-dir)."
        )
        print(
            "[warn] No PNGs in output dir — check ComfyUI console and node 132 (WAS Image Save)"
            + extra,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
