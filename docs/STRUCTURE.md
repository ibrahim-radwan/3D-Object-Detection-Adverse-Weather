# Project structure

This folder is the **canonical publishable home** for the foundation-model
adverse-weather 3D detection study:

```text
/home/wael/Test_Code/foundation-model-generated-adverse-weather-benchmark-for-3d-object-detection-robustness/
```

Heavy assets and detector day-to-day experiments stay outside this repo.
A sibling copy may also exist at `/home/wael/fm-adverse-weather-3d-bench/`;
prefer the `Test_Code` path above as the source of truth for toolkit code.

Former names: `weather-benchmark-toolkit`, `fm-adverse-weather-3d-bench`
(symlink `Test_Code/weather-benchmark-toolkit` still redirects here).

## Unified publishable packages (this repo)

| Path | Role |
|------|------|
| [`../rgb_qwen/`](../rgb_qwen/) | Qwen Image Edit via ComfyUI — RGB weather synthesis |
| [`../pcn_lidar/`](../pcn_lidar/) | PCN-LiDAR — WeatherGen → constant-\(N\) corruption |
| [`RGB_QWEN.md`](RGB_QWEN.md) | RGB usage guide |
| [`PCN_LIDAR.md`](PCN_LIDAR.md) | PCN algorithm + batch guide |
| [`tables/`](tables/) | Index of TeX / report assets (pointers only) |

```text
foundation-model-generated-adverse-weather-benchmark-for-3d-object-detection-robustness/
  rgb_qwen/
  pcn_lidar/
  docs/
  configs/ scripts/ tools/ src/ paper/  ← placeholders for fuller eval release
```

## Local research root (not shipped)

Primary detector implementation and reports:

```text
/home/wael/Test_Code/tools/mmdetection3d/
```

Related sibling:

```text
/home/wael/Test_Code/weathergen/                 # WeatherGen generation stack
```

Do **not** copy `weather_gen_all/`, datasets, or generated dumps into this repo.

## Mapping: placeholders → live research paths

| Scaffold | Live location (examples) |
|----------|--------------------------|
| `configs/` | `/home/wael/Test_Code/tools/mmdetection3d/configs/` |
| `tools/` | `/home/wael/Test_Code/tools/mmdetection3d/tools/` (e.g. `evaluate_nuscenes_splits_realism.py`) |
| `scripts/` | Shell/Python runners under the mmdet3d root (`evaluate_weather_*.py`, `test_*_weather.sh`) |
| `paper/` / `docs/tables/` | `/home/wael/Test_Code/tools/mmdetection3d/data/kitti/validation/reports/` and `data/reports/` |
| Datasets | `/home/wael/Test_Code/tools/mmdetection3d/data/` (**do not copy into this repo**) |

## Intended publish slice (future)

When cutting a fuller public release, additionally copy only:

1. Small configs needed to reproduce tables
2. Evaluation / validation scripts (no `.bin` / `.pkl` / checkpoints)
3. TeX tables and short paragraphs under `paper/`

Do **not** vendor KITTI / nuScenes / Waymo / STF point clouds or WeatherGen dumps.
