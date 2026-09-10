# PCN-LiDAR documentation

## Motivation

WeatherGen (and similar models) can synthesize weathered LiDAR, but the point
count and geometry often drift from the clear scan. **PCN** (project constant-\(N\)
noise) keeps the clear scan’s cardinality and injects weather outliers mined
from the generated cloud in a severity-controlled way.

## Algorithm (summary)

1. **Align** generated xyz to original via multi-stage trimmed ICP (optional
   discrete axis-convention search).
2. **Mine noise** with residual \(d(x)=\min_y\|x-y\|\) and range-aware threshold
   \(\tau(r)=\tau_0 + k\,r\).
3. **Inject** a weather/severity-dependent fraction of points into the clear
   cloud (prefer far ranges), with anisotropic jitter and intensity
   attenuation / dropout. Output \(N\) equals original \(N\).

Supported `--weather` schedules: `clear`, `rain`, `fog`, `snow`.

## End-to-end batch

```text
clear .bin  →  (optional) Waymo6→KITTI4  →  WeatherGen generate.py
                                              ↓
                                         generated.bin
                                              ↓
                    inject_pcn_weather_lidar.py × {light, medium, heavy}
                                              ↓
                         velodyne_{weather}_{light|moderate|severe}/
```

See [`../pcn_lidar/README.md`](../pcn_lidar/README.md) for CLI flags and
WeatherGen setup.

## Toolkit modules

| File | Role |
|------|------|
| `pcn_lidar/inject_pcn_weather_lidar.py` | Single-frame PCN CLI + algorithm |
| `pcn_lidar/batch_weathergen_pcn_lidar.py` | Clear → WeatherGen → PCN batch |

## Originals (unchanged)

| Piece | Path |
|-------|------|
| PCN core | `Test_Code/tools/mmdetection3d/tools/project_rain_noise.py` |
| Batch | `Test_Code/tools/mmdetection3d/tools/batch_clear_to_weather_weathergen_pcn.py` |
| Legacy KITTI-only wrapper | `Test_Code/tools/mmdetection3d/tools/run_project_rain_noise.py` (stale `--mode` flags; prefer batch script) |
| WeatherGen checkout | `Test_Code/weathergen` |

## References

- WeatherGen: https://github.com/wuyang98/weathergen  
- MMDetection3D: https://github.com/open-mmlab/mmdetection3d  
