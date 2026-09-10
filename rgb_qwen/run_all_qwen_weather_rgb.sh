#!/usr/bin/env bash
# Sequential Qwen RGB generation over the weather taxonomy.
# Portable version of mmdetection3d/project/run_all_qwen_remaining_conditions.sh
#
# Usage:
#   INPUT_DIR=/path/to/clear/images OUT_ROOT=/path/to/out ./run_all_qwen_weather_rgb.sh
#   LIMIT=2 WEATHERS=fog,snow ./run_all_qwen_weather_rgb.sh
#
# Env:
#   COMFY_URL   default http://127.0.0.1:8188
#   LIMIT       optional max images per condition
#   WEATHERS    optional comma list (default: full day+night taxonomy)
#   INTENSITIES default "light medium heavy" for day; night* uses light only

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/run_qwen_weather_rgb.py"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
INPUT_DIR="${INPUT_DIR:-}"
OUT_ROOT="${OUT_ROOT:-}"
LIMIT="${LIMIT:-}"

if [[ -z "$INPUT_DIR" || -z "$OUT_ROOT" ]]; then
  echo "Usage: INPUT_DIR=... OUT_ROOT=... $0" >&2
  exit 1
fi

DAY_DEFAULT=(fog snow sandstorm rain_fog snow_fog snow_rain rain)
NIGHT_DEFAULT=(night night_fog night_rain night_snow)

if [[ -n "${WEATHERS:-}" ]]; then
  IFS=',' read -r -a ALL_W <<< "$WEATHERS"
else
  ALL_W=("${DAY_DEFAULT[@]}" "${NIGHT_DEFAULT[@]}")
fi

_run() {
  local weather="$1" intensity="$2"
  local out="$OUT_ROOT/$weather/$intensity"
  mkdir -p "$out"
  echo ">>> $weather/$intensity -> $out"
  local args=(
    --comfy-url "$COMFY_URL"
    --weather "$weather"
    --intensity "$intensity"
    --input-dir "$INPUT_DIR"
    --output-dir "$out"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  python -u "$PY" "${args[@]}"
}

for w in "${ALL_W[@]}"; do
  case "$w" in
    night|night_fog|night_rain|night_snow)
      _run "$w" light
      ;;
    *)
      for sev in ${INTENSITIES:-light medium heavy}; do
        _run "$w" "$sev"
      done
      ;;
  esac
done

echo "Done."
