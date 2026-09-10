"""Qwen Image Edit (ComfyUI) RGB weather synthesis helpers.

Public entry points (run as scripts)::

  run_qwen_weather_rgb.py         — single weather/intensity CLI
  run_all_qwen_weather_rgb.sh     — sequential multi-condition shell driver
  qwen_weather_prompts.py         — canonical prompt library (+ JSON twin)
  comfyui_http_client.py          — ComfyUI REST client
  workflow_qwen_image_edit.json   — ComfyUI API workflow graph
"""

from .comfyui_http_client import ComfyClient  # noqa: F401
from .qwen_weather_prompts import (  # noqa: F401
    NEGATIVE_PROMPT,
    WEATHER_PROMPTS,
    get_negative_prompt,
    get_prompt,
    list_intensities,
    list_weathers,
)

__all__ = [
    "ComfyClient",
    "NEGATIVE_PROMPT",
    "WEATHER_PROMPTS",
    "get_negative_prompt",
    "get_prompt",
    "list_intensities",
    "list_weathers",
]
