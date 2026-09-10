"""PCN-LiDAR: WeatherGen + constant-N noise projection helpers.

Public entry points (run as scripts)::

  inject_pcn_weather_lidar.py   — single-frame PCN injection CLI
  batch_weathergen_pcn_lidar.py — WeatherGen → PCN batch CLI

Library helpers are imported lazily so ``python -m pcn_lidar.inject_pcn_weather_lidar``
does not double-load the CLI module via this package ``__init__``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "inject_noise_constantN_weather",
    "load_bin",
    "params_from_weather",
    "save_bin",
    "severity_scalar",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import inject_pcn_weather_lidar as _core

        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
