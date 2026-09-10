# RGB weather synthesis (Qwen + ComfyUI)

## What this does

Applies **Qwen Image Edit** through a local **ComfyUI** server to turn clear
driving images into weather-conditioned RGB (fog, rain, snow, night, mixes, …)
while trying to preserve object geometry via a light post-process blend.

## Prerequisites

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running (default `http://127.0.0.1:8188`).
2. Qwen Image Edit graph / custom nodes compatible with
   `rgb_qwen/workflow_qwen_image_edit.json` (nodes referenced by the runner:
   `115:111` / `115:110` prompts, `134` Load Image Batch, `132` WAS Image Save).
3. Python deps: `requests`, `numpy`, `Pillow`, `tqdm`.

## Prompts

All strings are centralized in
[`../rgb_qwen/qwen_weather_prompts.py`](../rgb_qwen/qwen_weather_prompts.py):

| Weather keys | Intensities |
|--------------|-------------|
| `rain`, `fog`, `snow`, `sandstorm`, `rain_fog`, `snow_fog`, `snow_rain` | `light`, `medium`, `heavy` |
| `night`, `night_fog`, `night_rain`, `night_snow` | `light` only |

Shared negative prompt: `NEGATIVE_PROMPT` / `get_negative_prompt()`.

`qwen_weather_prompts.json` is a JSON dump of the same content for tools that prefer files.
Regenerate it with `python qwen_weather_prompts.py`.

## Run one condition

```bash
cd rgb_qwen
python run_qwen_weather_rgb.py \
  --weather rain --intensity heavy \
  --input-dir /path/to/clear/image_0 \
  --output-dir /path/to/rain/heavy \
  --comfy-url http://127.0.0.1:8188
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--dry-run` | Print paths / prompt preview; no Comfy calls |
| `--limit N` | Smoke-test on first N images |
| `--staging-dir DIR` | WAS writes here first, then move to `--output-dir` |
| `--prompts PATH` | Override with an external prompts JSON |
| `--workflow PATH` | Alternate Comfy API workflow JSON |

Optional env: `COMFY_QWEN_STAGING` (same as `--staging-dir`).

## Batching multiple conditions

```bash
INPUT_DIR=/path/to/clear OUT_ROOT=/path/to/out \
  ./run_all_qwen_weather_rgb.sh
# optional: LIMIT=2 WEATHERS=fog,snow COMFY_URL=http://127.0.0.1:8188
```

## Toolkit modules

| File | Role |
|------|------|
| `run_qwen_weather_rgb.py` | CLI entry for one weather/intensity |
| `comfyui_http_client.py` | ComfyUI REST helpers |
| `qwen_weather_prompts.py` | Canonical positive/negative prompts |
| `qwen_weather_prompts.json` | JSON twin of the prompt module |
| `workflow_qwen_image_edit.json` | ComfyUI API workflow |
| `run_all_qwen_weather_rgb.sh` | Sequential multi-condition driver |

## Originals (unchanged)

| Piece | Path under `Test_Code/tools/mmdetection3d/project/` |
|-------|-----------------------------------------------------|
| Runner | `run_qwen_weather.py` |
| Prompts | `prompts.json` |
| Workflow | `02_qwen_Image_edit_subgraphed.json`, `_api.json` |
| Batch shell | `run_all_qwen_remaining_conditions.sh` |
| Older hard-coded scripts | `script.py`, `script_nuscenes.py` |
