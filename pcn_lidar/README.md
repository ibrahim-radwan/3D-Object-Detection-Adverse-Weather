# PCN-LiDAR

Constant-\(N\) weather noise projection for LiDAR, typically fed by
**WeatherGen** synthetic scans.

## Official dependencies (not vendored)

| Dependency | Link | Role |
|------------|------|------|
| WeatherGen | https://github.com/wuyang98/weathergen | Generate weathered LiDAR (`generate.py`) |
| MMDetection3D (optional) | https://github.com/open-mmlab/mmdetection3d | Downstream 3D detection eval |

Set the WeatherGen checkout via:

```bash
export WEATHERGEN_ROOT=/path/to/weathergen
# or
python batch_weathergen_pcn_lidar.py --weathergen-root /path/to/weathergen ...
```

Default checkpoint path inside WeatherGen:

```text
$WEATHERGEN_ROOT/checkpoint/diffusion_0000100000.pth
```

## PCN steps (single frame)

1. Start from clear KITTI-style `.bin` (`x,y,z,intensity`, float32) as `--orig`.
2. Obtain a weathered scan `--gen` (WeatherGen `generated.bin`, or any comparable source).
3. Run PCN:

```bash
python inject_pcn_weather_lidar.py \
  --orig clear.bin \
  --gen generated.bin \
  --out projected.bin \
  --weather rain \
  --severity medium \
  --prefer_far
```

What PCN does:

1. Align `--gen` to `--orig` (trimmed multi-stage ICP; optional axis search).
2. Mine weather outliers via range-aware residual threshold.
3. Replace a severity-dependent fraction of clear points with mined noise
   (jitter + intensity attenuation/dropout), **keeping the same point count**.

## Batch (WeatherGen → PCN)

```bash
python batch_weathergen_pcn_lidar.py \
  --kitti-clear-lidar /data/kitti/validation/velodyne \
  --kitti-out-root /data/kitti/validation \
  --weathers rain,fog,snow
```

Also supports `--waymo-root` and `--nuscenes-root` layouts with `clear/{velodyne|lidar}/`.

Folder naming:

| WeatherGen weather | Output folders (KITTI flat) |
|--------------------|-----------------------------|
| rain / fog / snow  | `velodyne_{weather}_{light\|moderate\|severe}/` |

Severity mapping: folder `moderate` → PCN `--severity medium`, `severe` → `heavy`.

Cache (safe to delete): `_weathergen_pcn_cache/<dataset>/<weather>/`.

## Files

| File | Purpose |
|------|---------|
| `inject_pcn_weather_lidar.py` | Core PCN algorithm + single-frame CLI |
| `batch_weathergen_pcn_lidar.py` | Orchestrates WeatherGen → PCN for a split |
| `__init__.py` | Re-exports core helpers for library use |
