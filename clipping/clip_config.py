"""
clip_config — Per-Clip Configuration Resolver

Merges global CLI config with per-clip overrides so every rank/clip can have
its own rendering behaviour (ratio, broll, split-screen, hook, bgm, subs,
font style, face detector, quality presets, etc.).

Override precedence (highest → lowest):
  1. External JSON file (--clip-config) keyed by rank
  2. Clip metadata returned by AI (clip dict itself)
  3. Global CLI flags / defaults (cfg)
"""

import json
import os
from copy import deepcopy
from types import SimpleNamespace


# Fields that a clip (or external JSON) may override.
# Each entry: (target_attr, default_source, transformer)
# target_attr   = name on the resolved SimpleNamespace
# default_source= callable(cfg) that returns the global default
# transformer   = optional function(raw_value) -> typed_value
_CLIP_OVERRIDES = {
    # ── Content toggles ──
    "use_broll":        ("use_broll",        lambda c: c.use_broll,        bool),
    "use_hook_glitch":  ("use_hook_glitch",  lambda c: c.use_hook_glitch,  bool),
    "use_auto_bgm":     ("use_auto_bgm",     lambda c: c.use_auto_bgm,     bool),
    "no_subs":          ("no_subs",          lambda c: c.no_subs,          bool),
    "use_karaoke_effect":("use_karaoke_effect",lambda c: c.use_karaoke_effect, bool),
    "use_advanced_text":("use_advanced_text",lambda c: c.use_advanced_text, bool),
    "use_advanced_text_on_hook": ("use_advanced_text_on_hook", lambda c: c.use_advanced_text_on_hook, bool),
    # ── Layout / Mode ──
    "use_split_screen": ("use_split_screen", lambda c: c.use_split_screen, bool),
    "use_camera_switch":("use_camera_switch",lambda c: c.use_camera_switch, bool),
    "split_trigger":    ("split_trigger",    lambda c: c.split_trigger,    str),
    "split_auto_zoom":  ("split_auto_zoom",  lambda c: c.split_auto_zoom,  bool),
    "split_zoom":       ("split_zoom",       lambda c: c.split_zoom,       float),
    "split_v_align":    ("split_v_align",    lambda c: c.split_v_align,    float),
    "split_max_zoom":   ("split_max_zoom",   lambda c: c.split_max_zoom,   float),
    "static_crop":      ("static_crop",      lambda c: c.static_crop,      bool),
    # ── Geometry ──
    "ratio":            ("selected_ratio",    lambda c: c.selected_ratio,    str),
    "hook_duration":    ("hook_duration",      lambda c: c.hook_duration,      int),
    "words_per_sub":    ("max_words_per_subtitle", lambda c: c.max_words_per_subtitle, int),
    # ── Typography ──
    "font_style":       ("active_font_style",  lambda c: c.active_font_style,  str),
    # ── Face Detection ──
    "face_detector":    ("face_detector",    lambda c: c.face_detector,    str),
    "yolo_size":        ("yolo_size",        lambda c: c.yolo_size,        str),
    # ── Video Quality ──
    "video_preset":     ("video_preset",     lambda c: c.video_preset,     str),
    "video_scale_algo": ("video_scale_algo", lambda c: c.video_scale_algo,  str),
    "video_sharpen":    ("video_sharpen",    lambda c: c.video_sharpen,     bool),
    "video_cq":         ("video_quality_cq", lambda c: c.video_quality_cq,  int),
    "video_crf":        ("video_quality_crf",lambda c: c.video_quality_crf, int),
    "video_bitrate":    ("video_bitrate",    lambda c: c.video_bitrate,     str),
    # ── Whisper ──
    "whisper_model":    ("whisper_model",    lambda c: c.whisper_model,    str),
    "whisper_device":   ("whisper_device",   lambda c: c.whisper_device,   str),
    "whisper_compute_type": ("whisper_compute_type", lambda c: c.whisper_compute_type, str),
    # ── Transcription ──
    "use_yt_transcript_api": ("use_yt_transcript_api", lambda c: c.use_yt_transcript_api, bool),
    # ── AI ──
    "ai_provider":      ("ai_provider",      lambda c: c.ai_provider,      str),
    "gemini_model":     ("gemini_model",     lambda c: c.gemini_model,     str),
    "ollama_url":       ("ollama_url",       lambda c: c.ollama_url,       str),
    "ollama_model":     ("ollama_model",     lambda c: c.ollama_model,     str),
    "ollama_fallback_url":  ("ollama_fallback_url",  lambda c: c.ollama_fallback_url,  str),
    "ollama_fallback_model":("ollama_fallback_model",lambda c: c.ollama_fallback_model,str),
}


def _coerce(val, transformer):
    if transformer is None or val is None:
        return val
    try:
        return transformer(val)
    except Exception:
        return val


def _pick_from_sources(keys, clip_dict, external_dict, cfg):
    """Return first non-None value found in keys order."""
    for k in keys:
        if external_dict is not None and k in external_dict and external_dict[k] is not None:
            return external_dict[k]
        if clip_dict is not None and k in clip_dict and clip_dict[k] is not None:
            return clip_dict[k]
    return None


def load_external_clip_config(path: str | None) -> dict:
    """Load a JSON file keyed by rank (as string) → override dict."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️ clip-config JSON root must be an object, got {type(data).__name__}. Ignored.")
            return {}
        return data
    except Exception as e:
        print(f"⚠️ Failed to load clip-config from {path}: {e}. Ignored.")
        return {}


def resolve_clip_cfg(clip: dict, cfg: SimpleNamespace, external: dict | None = None) -> SimpleNamespace:
    """
    Create a per-clip SimpleNamespace that inherits global cfg but applies overrides.

    Parameters
    ----------
    clip : dict
        Individual clip metadata (usually from AI analysis).
    cfg : SimpleNamespace
        Global runtime configuration.
    external : dict, optional
        External overrides keyed by rank string, e.g. {"1": {"ratio": "1:1"}}.

    Returns
    -------
    SimpleNamespace
        A new config object with per-clip overrides applied.
    """
    rank = str(clip.get("rank", "unknown"))
    ext = external.get(rank, {}) if external else {}

    # Deep-copy cfg to avoid mutating global state
    clip_cfg = SimpleNamespace(**vars(cfg))

    # Apply known overrides
    for alias, (target_attr, default_fn, transformer) in _CLIP_OVERRIDES.items():
        # Allow both positive and negative aliases in JSON/clip dict
        aliases = [alias]
        if alias.startswith("no_"):
            # e.g. no_subs  ↔  use_subs
            aliases.append(alias.replace("no_", "use_", 1))
        elif alias.startswith("use_"):
            # e.g. use_broll ↔ no_broll
            aliases.append(alias.replace("use_", "no_", 1))

        raw = _pick_from_sources(aliases, clip, ext, cfg)
        if raw is not None:
            coerced = _coerce(raw, transformer)
            # Handle boolean flips for no_ / use_ pairs
            if isinstance(coerced, bool) and alias.startswith("no_"):
                coerced = not coerced
            setattr(clip_cfg, target_attr, coerced)

    # ── Special convenience aliases ──
    # If user passes "ratio" in JSON it maps to selected_ratio (already handled above)
    # If user passes "render_height" it maps to render_output_height
    if ext.get("render_height") is not None or clip.get("render_height") is not None:
        rh = ext.get("render_height") if ext.get("render_height") is not None else clip.get("render_height")
        clip_cfg.render_output_height = str(rh)

    # If user passes "source_height" in JSON/clip it maps to download_source_height
    if ext.get("source_height") is not None or clip.get("source_height") is not None:
        sh = ext.get("source_height") if ext.get("source_height") is not None else clip.get("source_height")
        clip_cfg.download_source_height = sh

    # If user passes "broll_list" explicitly in external JSON, replace AI's list entirely
    if ext.get("broll_list") is not None:
        clip["broll_list"] = ext["broll_list"]
    if ext.get("typography_plan") is not None:
        clip["typography_plan"] = ext["typography_plan"]
    if ext.get("bgm_mood") is not None:
        clip["bgm_mood"] = ext["bgm_mood"]
    if ext.get("keep_segments") is not None:
        clip["keep_segments"] = ext["keep_segments"]

    # Print what changed (only if something actually differs)
    changes = []
    for alias, (target_attr, default_fn, _) in _CLIP_OVERRIDES.items():
        old = getattr(cfg, target_attr, None)
        new = getattr(clip_cfg, target_attr, None)
        if old != new:
            changes.append(f"{target_attr}: {old} → {new}")
    if changes:
        print(f"   🎛️ Clip overrides applied (Rank {rank}): {', '.join(changes)}")

    return clip_cfg


def print_available_overrides():
    """Print a help table of all overridable fields."""
    print("Per-clip overridable fields (use in --clip-config JSON or AI metadata):")
    for alias, (target_attr, _, transformer) in _CLIP_OVERRIDES.items():
        tname = transformer.__name__ if transformer else "any"
        print(f"   {alias:<25} → {target_attr:<30} ({tname})")
