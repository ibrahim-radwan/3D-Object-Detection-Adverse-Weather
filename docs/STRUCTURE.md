# Project structure

This repository is the publishable home for foundation-model adverse-weather
synthesis helpers used in the 3D detection robustness study.

Former names: `weather-benchmark-toolkit`, `fm-adverse-weather-3d-bench`.

## What ships here

| Path | Role |
|------|------|
| [`../rgb_qwen/`](../rgb_qwen/) | Qwen Image Edit via ComfyUI — RGB weather synthesis |
| [`../pcn_lidar/`](../pcn_lidar/) | PCN-LiDAR — WeatherGen → constant-\(N\) corruption |
| [`RGB_QWEN.md`](RGB_QWEN.md) | RGB usage guide |
| [`PCN_LIDAR.md`](PCN_LIDAR.md) | PCN algorithm + batch guide |
| [`tables/`](tables/) | Index for TeX / report assets (placeholders for now) |
| [`../configs/`](../configs/) | Placeholder for future detector / eval configs |

```text
foundation-model-generated-adverse-weather-benchmark-for-3d-object-detection-robustness/
  rgb_qwen/
  pcn_lidar/
  docs/
  configs/
  README.md
  LICENSE
  requirements.txt
```

## What is not shipped

- Dataset binaries (KITTI / nuScenes / Waymo / STF, etc.)
- WeatherGen checkpoints and generated point-cloud dumps
- ComfyUI / model weights
- Full MMDetection3D training and evaluation trees

Clone [WeatherGen](https://github.com/wuyang98/weathergen) and
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d) separately when
you need generation checkpoints or detector evaluation.

## Intended future additions

When cutting a fuller public release, this tree may additionally include:

1. Small configs under `configs/` needed to reproduce paper tables
2. Evaluation / validation scripts (no `.bin` / `.pkl` / checkpoints)
3. Curated TeX tables under `docs/tables/` (or a dedicated paper assets folder)

Do **not** vendor large sensor dumps or WeatherGen outputs into this repo.
