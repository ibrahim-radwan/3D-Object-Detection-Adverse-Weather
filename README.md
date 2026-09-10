# Foundation-Model-Generated Adverse Weather Benchmark for 3D Object Detection Robustness

Publishable helpers for the foundation-model (FM) adverse-weather benchmark
(RGB + LiDAR) used to study **3D object detection robustness**. This tree
copies the essential scripts from the research workspace under
`tools/mmdetection3d/` so originals keep working. It does **not** vendor
WeatherGen, MMDetection3D, ComfyUI, or dataset binaries.

> Formerly named `weather-benchmark-toolkit` / `fm-adverse-weather-3d-bench`.
> A symlink at `Test_Code/weather-benchmark-toolkit` → this folder is kept for convenience.

| Package | Role |
|---------|------|
| [`rgb_qwen/`](rgb_qwen/) | Qwen Image Edit via ComfyUI — RGB fog / rain / snow / night / … |
| [`pcn_lidar/`](pcn_lidar/) | PCN-LiDAR — project WeatherGen noise onto clear scans (constant \(N\)) |
| [`docs/`](docs/) | Longer notes for each modality |

## Paper

**Title:** Foundation-Model-Generated Adverse Weather Benchmark for 3D Object Detection Robustness

**Authors:**
Wael Issa¹, Weijian Deng², Mohammed Alotaibi¹, Karam M. Sallam³, Mohammed Hassanin¹, Abdelwahed Khamis⁴, Carlos Kuhn¹, and Ibrahim Radwan¹

**Affiliations:**
1. School of Information Technology & Systems, University of Canberra, Bruce, ACT 2617, Australia
2. Tsinghua Shenzhen International Graduate School, Tsinghua University, Shenzhen, China
3. Department of Computer Science, University of Sharjah, Sharjah, United Arab Emirates
4. Commonwealth Scientific and Industrial Research Organisation (CSIRO), Brisbane, Australia

**Corresponding author:** Ibrahim Radwan ([Ibrahim.Radwan@canberra.edu.au](mailto:Ibrahim.Radwan@canberra.edu.au))

**Funding:** This work was supported by the Australian Government’s National Road Safety Program.

**Journal:** [IEEE Open Journal of Intelligent Transportation Systems (OJ-ITS)](https://ieee-itss.org/pub/oj-its/) — submitted / under review.

## Citation

```bibtex
@article{issa2026fm_adverse_weather_3d,
  title   = {Foundation-Model-Generated Adverse Weather Benchmark for 3D Object Detection Robustness},
  author  = {Issa, Wael and Deng, Weijian and Alotaibi, Mohammed and Sallam, Karam M. and Hassanin, Mohammed and Khamis, Abdelwahed and Kuhn, Carlos and Radwan, Ibrahim},
  journal = {IEEE Open Journal of Intelligent Transportation Systems},
  year    = {2026},
  note    = {Submitted / under review}
}
```

## Install

```bash
cd foundation-model-generated-adverse-weather-benchmark-for-3d-object-detection-robustness
pip install -r requirements.txt   # requests, numpy, pillow, tqdm, scipy
```

Optional: `scikit-learn` only if SciPy’s `cKDTree` is unavailable (PCN NN fallback).

Add the project root to `PYTHONPATH` if you import packages or use
`python -m …` from another working directory:

```bash
export PYTHONPATH="/path/to/foundation-model-generated-adverse-weather-benchmark-for-3d-object-detection-robustness:${PYTHONPATH}"
```

---

## `rgb_qwen/` — RGB weather (Qwen + ComfyUI)

Synthesizes weather-conditioned camera images by queuing a ComfyUI API workflow
(Qwen Image Edit), then lightly post-processing each PNG to preserve object
geometry relative to the clear input.

### Prerequisites

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with Qwen Image Edit nodes
   compatible with `rgb_qwen/workflow_qwen_image_edit.json`.
2. ComfyUI listening (default `http://127.0.0.1:8188`).

### Modules / entry points

| Path | What it does | Inputs | Outputs | How to run |
|------|--------------|--------|---------|------------|
| `run_qwen_weather_rgb.py` | CLI: one weather + intensity over a folder of clear images | `--input-dir` images (`.jpg`/`.png`), `--weather`, `--intensity`, optional `--workflow` / `--prompts` | PNGs under `--output-dir` (same stem as input) | `python run_qwen_weather_rgb.py …` or `python -m rgb_qwen.run_qwen_weather_rgb …` |
| `comfyui_http_client.py` | Thin ComfyUI REST client (`/prompt`, `/queue`, `/history`, health) | Base URL | Library only (`ComfyClient`) | Imported by the runner |
| `qwen_weather_prompts.py` | Canonical positive/negative prompt strings + helpers | Weather key + intensity | Prompt strings / JSON serialization | Library; `python qwen_weather_prompts.py` regenerates the JSON twin |
| `qwen_weather_prompts.json` | JSON twin of `qwen_weather_prompts.py` | — | File for tools that prefer JSON | Used via `--prompts` or `--prompts-json-builtin` |
| `workflow_qwen_image_edit.json` | ComfyUI **API-format** workflow graph | Loaded by runner | — | Pass with `--workflow` if you fork it |
| `run_all_qwen_weather_rgb.sh` | Sequential batch over the weather taxonomy | Env: `INPUT_DIR`, `OUT_ROOT`; optional `LIMIT`, `WEATHERS`, `COMFY_URL`, `INTENSITIES` | `$OUT_ROOT/<weather>/<intensity>/` | `INPUT_DIR=… OUT_ROOT=… ./run_all_qwen_weather_rgb.sh` |
| `__init__.py` | Re-exports `ComfyClient`, prompt helpers | — | Package API | `from rgb_qwen import get_prompt, …` |

### Weather keys

| Keys | Intensities |
|------|-------------|
| `rain`, `fog`, `snow`, `sandstorm`, `rain_fog`, `snow_fog`, `snow_rain` | `light`, `medium`, `heavy` |
| `night`, `night_fog`, `night_rain`, `night_snow` | `light` only |

### Example (single condition)

```bash
cd rgb_qwen
python run_qwen_weather_rgb.py \
  --weather fog --intensity medium \
  --input-dir /path/to/clear/images \
  --output-dir /path/to/out/fog/medium \
  --comfy-url http://127.0.0.1:8188
```

Useful flags: `--dry-run` (no Comfy calls), `--limit N`, `--staging-dir DIR`
(or env `COMFY_QWEN_STAGING`), `--prompts PATH`, `--workflow PATH`.

### Example (all conditions)

```bash
INPUT_DIR=/path/to/clear OUT_ROOT=/path/to/out \
  LIMIT=2 ./run_all_qwen_weather_rgb.sh
```

More detail: [docs/RGB_QWEN.md](docs/RGB_QWEN.md).

---

## `pcn_lidar/` — LiDAR weather (WeatherGen + PCN)

**PCN** (project constant-\(N\) noise) keeps clear-scan cardinality and injects
weather outliers mined from a WeatherGen (or similar) generated cloud, with
severity-controlled replace / jitter / intensity attenuation / dropout.

### Prerequisites

1. Clone [WeatherGen](https://github.com/wuyang98/weathergen) and download its checkpoint.
2. Set `WEATHERGEN_ROOT` or pass `--weathergen-root` (batch script also probes
   sibling `../weathergen` under `Test_Code/` when present).

Default checkpoint: `$WEATHERGEN_ROOT/checkpoint/diffusion_0000100000.pth`.

### Modules / entry points

| Path | What it does | Inputs | Outputs | How to run |
|------|--------------|--------|---------|------------|
| `inject_pcn_weather_lidar.py` | Single-frame PCN: align → mine residual outliers → constant-\(N\) inject | `--orig` clear `.bin`, `--gen` weathered `.bin`, `--weather`, `--severity` | `--out` `.bin` with same \(N\) as clear (KITTI float32 `xyz+i`) | `python inject_pcn_weather_lidar.py …` or `python -m pcn_lidar.inject_pcn_weather_lidar …` |
| `batch_weathergen_pcn_lidar.py` | Batch: clear → (optional) Waymo6→KITTI4 → WeatherGen → PCN × {light,medium,heavy} | `--kitti-clear-lidar` / `--waymo-root` / `--nuscenes-root`, `--weathers`, WeatherGen root/ckpt | KITTI flat `velodyne_{weather}_{light\|moderate\|severe}/` or split folders `{weather}_{sev}/{lidar\|velodyne}/`; cache under `_weathergen_pcn_cache/` | `python batch_weathergen_pcn_lidar.py …` |
| `__init__.py` | Re-exports `load_bin`, `save_bin`, `params_from_weather`, `inject_noise_constantN_weather`, … | — | Package API | `from pcn_lidar import load_bin, …` |

Severity folder map: `light` → PCN `light`; `moderate` → PCN `medium`; `severe` → PCN `heavy`.

### Example (single frame)

```bash
cd pcn_lidar
python inject_pcn_weather_lidar.py \
  --orig clear.bin --gen weathergen_generated.bin --out out.bin \
  --weather rain --severity medium --prefer_far
```

### Example (KITTI validation batch)

```bash
export WEATHERGEN_ROOT=/path/to/weathergen
cd pcn_lidar
python batch_weathergen_pcn_lidar.py \
  --kitti-clear-lidar /path/to/validation/velodyne \
  --kitti-out-root /path/to/validation \
  --weathers rain,fog,snow \
  --limit 2
```

Smoke / reuse flags: `--skip-weathergen`, `--skip-pcn`, `--limit N`,
`--inject-pcn PATH` (alias `--project-noise` kept for older scripts).

Package notes: [pcn_lidar/README.md](pcn_lidar/README.md), [docs/PCN_LIDAR.md](docs/PCN_LIDAR.md).

---

## Layout

```
foundation-model-generated-adverse-weather-benchmark-for-3d-object-detection-robustness/
  README.md
  LICENSE
  requirements.txt
  .gitignore
  rgb_qwen/
    __init__.py
    run_qwen_weather_rgb.py          # CLI entry (one condition)
    run_all_qwen_weather_rgb.sh      # multi-condition shell driver
    comfyui_http_client.py           # ComfyUI REST helpers
    qwen_weather_prompts.py          # canonical prompts
    qwen_weather_prompts.json        # JSON twin
    workflow_qwen_image_edit.json    # ComfyUI API workflow
  pcn_lidar/
    __init__.py
    inject_pcn_weather_lidar.py      # PCN core + single-frame CLI
    batch_weathergen_pcn_lidar.py    # WeatherGen → PCN batch
    README.md
  docs/
    RGB_QWEN.md
    PCN_LIDAR.md
    STRUCTURE.md
    tables/                          # TeX / report index (pointers)
  configs/ scripts/ tools/ src/ paper/  # placeholders for fuller eval release
```

---

## Upstream originals (still authoritative for full experiments)

| Component | Original path |
|-----------|---------------|
| Qwen runner | `Test_Code/tools/mmdetection3d/project/run_qwen_weather.py` |
| Prompts JSON | `Test_Code/tools/mmdetection3d/project/prompts.json` |
| Batch shell | `Test_Code/tools/mmdetection3d/project/run_all_qwen_remaining_conditions.sh` |
| Comfy workflow | `Test_Code/tools/mmdetection3d/project/02_qwen_Image_edit_subgraphed*.json` |
| PCN core | `Test_Code/tools/mmdetection3d/tools/project_rain_noise.py` |
| PCN batch | `Test_Code/tools/mmdetection3d/tools/batch_clear_to_weather_weathergen_pcn.py` |
| WeatherGen | `Test_Code/weathergen` → https://github.com/wuyang98/weathergen |

Do not commit large point clouds, RGB dumps, or orchestration caches into this
project (see `.gitignore`).

## License

Released under the [MIT License](LICENSE).
Upstream MMDetection3D and third-party weather baselines retain their own licenses.
