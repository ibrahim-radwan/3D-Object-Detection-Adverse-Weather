"""Centralized Qwen Image Edit weather prompts for RGB weather synthesis.

All positive/negative prompt strings used by the ComfyUI runner live here.
``qwen_weather_prompts.json`` in this package is the serialized twin of this
module and is kept for workflow tooling that prefers JSON.

Condition keys match the dataset taxonomy used elsewhere in the benchmark:
  rain, fog, snow, sandstorm, rain_fog, snow_fog, snow_rain,
  night, night_fog, night_rain, night_snow

Intensity tiers: ``light`` | ``medium`` | ``heavy``
(night* conditions currently expose only ``light``.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

Intensity = str  # "light" | "medium" | "heavy"
WeatherName = str

# Global negative prompt shared across all weather conditions.
NEGATIVE_PROMPT: str = (
    "do not change any objects, geometry, layout, camera, or framing. "
    "do not add, remove, replace, hide, or alter any object. "
    "do not stylize, enhance, beautify, or hallucinate details. "
    "do not regenerate the scene. "
    "do not alter materials, colors, or textures except for realistic weather interaction. "
    "do not change lighting style or time of day. "
    "avoid any modification beyond weather effects. "
    "preserve the original scene exactly."
)

# ---------------------------------------------------------------------------
# Day / adverse weather
# ---------------------------------------------------------------------------

RAIN_LIGHT: str = (
    "Apply very light realistic rain.\n"
    "Add light natural rain streaks, wet surface reflections, and light ground mist.\n"
    "Do not alter any object shape, color, texture, or position."
)
RAIN_MEDIUM: str = (
    "Apply medium level realistic rain.\n"
    "Add medium level natural rain streaks, wet surface reflections, and light ground mist. "
    "Slightly reduce the visibilty using rain and fog. \n"
    "Do not alter any object shape, color, texture, or position."
)
RAIN_HEAVY: str = (
    "Apply heavy realistic rain with dense fog.\n"
    "Add natural rain streaks, wet surface reflections, and light ground mist.\n"
    "Reduce distance visibility using rain and fog haze only.\n"
    "Do not alter any object shape, color, texture, or position."
)

SNOW_LIGHT: str = (
    "Transform the scene to light snowfall. Remove sunlight and warm tones. "
    "Use an overcast sky with soft, flat lighting. Add sparse, gently falling snowflakes. "
    "Do not change any objects, shapes, colors, textures, or layout."
)
SNOW_MEDIUM: str = (
    "Transform the scene to moderate snowfall. Remove sunlight and warm tones. "
    "Use a fully overcast sky with flat, muted lighting. "
    "Add a dense layer of falling snow that slightly reduces visibility. "
    "Do not change any objects, shapes, colors, textures, or layout."
)
SNOW_HEAVY: str = (
    "Transform the scene to heavy snowfall. Remove sunlight and warm tones. "
    "Use a dark overcast sky with flat, low-contrast lighting. "
    "Add thick, swirling snow that strongly reduces visibility. "
    "Do not change any objects, shapes, colors, textures, or layout."
)

FOG_LIGHT: str = (
    "Add a very subtle, uniform atmospheric haze to imply light fog. "
    "Ensure all original object colors, shapes, textures, and positions remain perfectly "
    "preserved and visibly unchanged. Maintain the exact composition, viewpoint, "
    "resolution, and detail. Do not introduce new elements."
)
FOG_MEDIUM: str = (
    "Add light fogy weather. Maintain the original viewpoint, resolution, and detail level. "
    "Do not introduce any new structures or elements."
)
FOG_HEAVY: str = (
    "Add a heavy, uniform atmospheric haze to imply heavy fog. "
    "Ensure all original object colors, shapes, textures, and positions remain perfectly "
    "preserved and visibly unchanged. Maintain the exact composition, viewpoint, "
    "resolution, and detail. Do not introduce new elements."
)

RAIN_FOG_LIGHT: str = (
    "Apply very light realistic rain mixed with light fog.\n"
    "Add light natural rain streaks, subtle wet surface reflections, and a thin uniform fog layer. "
    "Slightly soften distant visibility using rain and fog only.\n"
    "Do not alter any object shape, color, texture, or position."
)
RAIN_FOG_MEDIUM: str = (
    "Apply medium level realistic rain mixed with fog.\n"
    "Add medium level natural rain streaks, wet surface reflections, light ground mist, "
    "and noticeable fog. Reduce mid-range visibility using rain and fog haze only.\n"
    "Do not alter any object shape, color, texture, or position."
)
RAIN_FOG_HEAVY: str = (
    "Apply heavy realistic rain with dense fog.\n"
    "Add natural rain streaks, wet surface reflections, and light ground mist. "
    "Reduce distance visibility using rain and fog haze only.\n"
    "Do not alter any object shape, color, texture, or position."
)

SNOW_RAIN_LIGHT: str = (
    "Transform the scene to light mixed snow and rain weather.\n"
    "Replace prior clear conditions with overcast, cool-toned lighting, light rain streaks, "
    "sparse snowflakes, and lightly damp surfaces. Maintain the exact same objects, geometry, "
    "colors, textures, viewpoint, and composition as the original."
)
SNOW_RAIN_MEDIUM: str = (
    "Transform the scene to moderate mixed snow and rain weather.\n"
    "Replace prior weather cues with overcast sky, cool diffuse lighting, steady rainfall "
    "mixed with visible snowfall, and wet surfaces. Preserve all original shapes, colors, "
    "textures, positions, geometry, and composition exactly."
)
SNOW_RAIN_HEAVY: str = (
    "Transform the scene to heavy mixed snow and rain weather.\n"
    "Replace prior sun, dry ground, and warm tones with deep overcast lighting, dense rain "
    "and snow mixture, and reduced atmospheric visibility. All original objects, colors, "
    "shapes, and textures must remain completely untouched and visible."
)

SNOW_FOG_LIGHT: str = (
    "Transform the scene to light snowy weather with fog.\n"
    "Replace clear conditions with cool overcast lighting, gently falling snow, and a thin "
    "uniform fog layer that softens distant visibility. Maintain the exact same objects, "
    "geometry, colors, textures, viewpoint, and composition."
)
SNOW_FOG_MEDIUM: str = (
    "Transform the scene to moderate snowfall with fog.\n"
    "Replace prior weather conditions with overcast sky, cool diffuse lighting, steady "
    "snowfall, and noticeable fog that reduces mid-range visibility. Preserve all original "
    "objects, shapes, colors, textures, and positions exactly."
)
SNOW_FOG_HEAVY: str = (
    "Transform the scene to heavy snowfall with dense fog.\n"
    "Replace prior sun, dry ground, and warm tones with deep overcast lighting, dense "
    "swirling snowflakes, and thick fog that obscures depth. All original objects, colors, "
    "shapes, and textures must remain completely untouched and visible."
)

SANDSTORM_LIGHT: str = (
    "Add a very subtle, uniform haze of fine, semi-transparent airborne yellow dust to the scene. "
    "Use atmospheric perspective to imply depth, with a very gentle reduction in clarity for "
    "distant objects, similar to a light fog effect. Ensure all original object colors, shapes, "
    "textures, and positions remain perfectly preserved and visibly unchanged. Maintain the exact "
    "composition, viewpoint, resolution, and detail. Do not introduce new elements."
)
SANDSTORM_MEDIUM: str = (
    "Apply light dust or sandstorm conditions, depicted strictly as a volumetric layer of fine, "
    "semi-transparent airborne particles with a warm, sandy tone. Simulate the depth and "
    "diffusion of fog, causing mid-ground details to soften slightly and distant objects to "
    "become faintly obscured. Maintain the original viewpoint, resolution, and detail level. "
    "Do not introduce any new structures or elements."
)
SANDSTORM_HEAVY: str = (
    "Add a heavy, uniform haze of dense, fine yellow sand, visualized strictly as "
    "semi-transparent airborne particles with pronounced depth attenuation. Use strong "
    "atmospheric perspective, like in dense fog, where foreground objects are clearer, "
    "mid-ground is softened, and background elements are significantly faded and muted. "
    "Ensure all original object colors, shapes, textures, and positions remain perfectly "
    "preserved and visibly unchanged. Maintain the exact composition, viewpoint, resolution, "
    "and detail. Do not introduce new elements."
)

# ---------------------------------------------------------------------------
# Night (lighting change; intensity tier is ``light`` only in current taxonomy)
# ---------------------------------------------------------------------------

NIGHT_LIGHT: str = (
    "Transform the scene to night-time while preserving all objects and layout. "
    "Remove all previous daytime weather cues such as sunlight, bright sky, and hard shadows. \n"
    "Replace them with realistic night illumination: low ambient light, cool tones, car "
    "headlights and streetlights as primary light sources, and a dark sky. Do not modify the "
    "resolution, texture density, or introduce new scene objects; only update the lighting "
    "and atmosphere."
)

NIGHT_FOG_LIGHT: str = (
    "Transform the scene to a misty night-time environment while preserving all objects and layout. "
    "Remove all previous daytime cues like sunlight, bright sky, and hard shadows. Replace them "
    "with a dark night sky, very low cool ambient light, and soft illumination only from existing "
    "light sources such as streetlights and car headlights. Add a slight, uniform fog that creates "
    "subtle halos around lights and softens depth, without obscuring details. Do not modify "
    "resolution, texture density, or introduce any new objects; only update the lighting, sky, "
    "and atmosphere."
)

NIGHT_RAIN_LIGHT: str = (
    "Transform the scene to a rainy night-time environment while preserving all objects and layout. "
    "Remove all previous daytime cues like sunlight, bright sky, and hard shadows. Replace them "
    "with a dark night sky, very low cool ambient light, and soft illumination only from existing "
    "light sources such as streetlights and car headlights. Add light rain throughout the scene, "
    "visible as streaks in the air and subtle wet reflections on surfaces. Do not modify "
    "resolution, texture density, or introduce any new objects; only update the lighting, sky, "
    "and atmosphere."
)

NIGHT_SNOW_LIGHT: str = (
    "Transform the scene to a snowy night-time environment while preserving all objects and layout. "
    "Remove all previous daytime cues like sunlight, bright sky, and hard shadows. Replace them "
    "with a dark night sky, very low cool ambient light, and soft illumination only from existing "
    "light sources such as streetlights and car headlights. Add light snowfall throughout the "
    "scene, with sparse snowflakes drifting gently through the air. Do not modify resolution, "
    "texture density, or introduce any new objects; only update the lighting, sky, and atmosphere."
)

# Nested map: weather -> intensity -> positive prompt (API surface used by the runner).
WEATHER_PROMPTS: Dict[WeatherName, Dict[Intensity, str]] = {
    "rain": {"light": RAIN_LIGHT, "medium": RAIN_MEDIUM, "heavy": RAIN_HEAVY},
    "snow": {"light": SNOW_LIGHT, "medium": SNOW_MEDIUM, "heavy": SNOW_HEAVY},
    "fog": {"light": FOG_LIGHT, "medium": FOG_MEDIUM, "heavy": FOG_HEAVY},
    "rain_fog": {
        "light": RAIN_FOG_LIGHT,
        "medium": RAIN_FOG_MEDIUM,
        "heavy": RAIN_FOG_HEAVY,
    },
    "snow_rain": {
        "light": SNOW_RAIN_LIGHT,
        "medium": SNOW_RAIN_MEDIUM,
        "heavy": SNOW_RAIN_HEAVY,
    },
    "snow_fog": {
        "light": SNOW_FOG_LIGHT,
        "medium": SNOW_FOG_MEDIUM,
        "heavy": SNOW_FOG_HEAVY,
    },
    "sandstorm": {
        "light": SANDSTORM_LIGHT,
        "medium": SANDSTORM_MEDIUM,
        "heavy": SANDSTORM_HEAVY,
    },
    "night": {"light": NIGHT_LIGHT},
    "night_fog": {"light": NIGHT_FOG_LIGHT},
    "night_rain": {"light": NIGHT_RAIN_LIGHT},
    "night_snow": {"light": NIGHT_SNOW_LIGHT},
}


def list_weathers() -> list[str]:
    """Return sorted weather condition names."""
    return sorted(WEATHER_PROMPTS.keys())


def list_intensities(weather: str) -> list[str]:
    """Return intensity tiers available for ``weather``."""
    if weather not in WEATHER_PROMPTS:
        raise KeyError(f"Unknown weather '{weather}'. Known: {list_weathers()}")
    return list(WEATHER_PROMPTS[weather].keys())


def get_prompt(weather: str, intensity: str = "medium") -> str:
    """Return the positive prompt for ``weather`` / ``intensity``.

    Raises:
        KeyError: if the weather or intensity tier is missing.
    """
    try:
        return WEATHER_PROMPTS[weather][intensity]
    except KeyError as exc:
        known = list_intensities(weather) if weather in WEATHER_PROMPTS else list_weathers()
        raise KeyError(
            f"Missing prompt weather_prompts.{weather}.{intensity}; "
            f"available for this key: {known}"
        ) from exc


def get_negative_prompt() -> str:
    """Return the shared negative prompt."""
    return NEGATIVE_PROMPT


def as_config_dict() -> dict:
    """Serialize prompts to the on-disk ``qwen_weather_prompts.json`` shape."""
    return {
        "negative_prompt": NEGATIVE_PROMPT,
        "weather_prompts": {k: dict(v) for k, v in WEATHER_PROMPTS.items()},
    }


def load_prompts_json(path: str | Path) -> tuple[str, Mapping[str, Mapping[str, str]]]:
    """Load ``(negative_prompt, weather_prompts)`` from a prompts JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("negative_prompt", ""), data["weather_prompts"]


def write_prompts_json(path: str | Path) -> None:
    """Write the centralized prompts to ``path`` (pretty JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(as_config_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    # Refresh the packaged JSON twin from this module.
    out = Path(__file__).resolve().parent / "qwen_weather_prompts.json"
    write_prompts_json(out)
    print(f"Wrote {out}")
