#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_prompt_enhancer.py

A single-file prompt helper that turns:
  (user prompt + optional reference image + target video seconds) -> 
  a WAN 2.2–style enhanced prompt + time-coded scenario + camera, actor, lighting, tech notes.

Usage examples:
  python video_prompt_enhancer.py --prompt "a lone hiker on a ridge at sunrise" --seconds 5
  python video_prompt_enhancer.py --prompt "cyberpunk alley with neon signs" --image ./ref.jpg --seconds 7 --format text
  python video_prompt_enhancer.py --prompt "close-up portrait, studio" --image ./face.jpg --seconds 3 --seed 42 --fps 24 --ar 16:9

Dependencies (minimal):
  pip install pillow numpy

Optional (auto-caption if you want; skipped if not available and no internet):
  pip install transformers torch torchvision   # heavy; not required

Notes:
- No external API calls are required.
- Image analysis is heuristic: brightness, saturation, dominant hue (warm/cool), sharpness proxy, orientation.
- Output defaults to JSON; pass --format text for a human-readable block.

Inspired by WAN 2.2 prompt structure and a three-part build (caption/character, action/dynamics, camera/cinematography).
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# -------------------------------
# Utilities
# -------------------------------

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def pct(x: float) -> str:
    return f"{round(100.0 * x)}%"

def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None

def resize_thumb(im: Image.Image, max_side: int = 256) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    if s <= max_side:
        return im
    scale = max_side / float(s)
    return im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def rgb_to_hsv_arr(rgb_arr: np.ndarray) -> np.ndarray:
    # rgb_arr: HxWx3 in [0,255]
    rgb = rgb_arr.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin + 1e-6

    # Hue calculation
    hue = np.zeros_like(cmax)
    mask = (delta > 1e-6)
    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask
    hue[r_is_max] = ((g - b)[r_is_max] / delta[r_is_max]) % 6.0
    hue[g_is_max] = ((b - r)[g_is_max] / delta[g_is_max]) + 2.0
    hue[b_is_max] = ((r - g)[b_is_max] / delta[b_is_max]) + 4.0
    hue = hue * 60.0  # degrees

    # Saturation and Value
    sat = np.where(cmax < 1e-6, 0.0, delta / (cmax + 1e-6))
    val = cmax

    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv

def hue_name(deg: float) -> str:
    # Compact bins for narrative use
    d = deg % 360
    if (d < 15) or (d >= 345):
        return "red"
    if d < 45:
        return "orange"
    if d < 75:
        return "yellow"
    if d < 150:
        return "green"
    if d < 195:
        return "cyan"
    if d < 255:
        return "blue"
    if d < 315:
        return "purple"
    return "magenta"

def temp_from_hue(deg: float) -> str:
    d = deg % 360
    if d < 60 or d >= 330:
        return "warm"
    if 180 <= d <= 300:
        return "cool"
    return "neutral"

def bucketize(value: float, cuts: Tuple[float, float]) -> str:
    lo, hi = cuts
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "medium"

# -------------------------------
# Image analysis
# -------------------------------

@dataclass
class ImageAnalysis:
    width: int
    height: int
    orientation: str
    mean_brightness: float
    mean_saturation: float
    contrast: float
    sharpness: float
    dom_hue_deg: Optional[float]
    dom_hue_name: Optional[str]
    dom_temp: Optional[str]

def analyze_image(im: Image.Image) -> ImageAnalysis:
    w, h = im.size
    orientation = "landscape" if w >= h else "portrait"
    im_small = resize_thumb(im, 320)
    arr = np.asarray(im_small)
    hsv = rgb_to_hsv_arr(arr)
    # Basic stats
    mean_brightness = float(np.mean(hsv[..., 2]))
    mean_saturation = float(np.mean(hsv[..., 1]))
    # Contrast ~ std of V
    contrast = float(np.std(hsv[..., 2]))
    # Dominant hue from moderately saturated pixels
    mask = hsv[..., 1] > 0.2
    hue_vals = hsv[..., 0][mask]
    dom_hue_deg = float(hue_vals.mean()) if hue_vals.size > 0 else None
    dom_name = hue_name(dom_hue_deg) if dom_hue_deg is not None else None
    dom_temp = temp_from_hue(dom_hue_deg) if dom_hue_deg is not None else None

    # Sharpness proxy: average gradient magnitude variance on grayscale
    gray = np.asarray(im_small.convert("L")).astype(np.float32) / 255.0
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    grad_mag = 0.5 * (gx.mean() + gy.mean())
    sharpness = float(np.var(gx) + np.var(gy) + grad_mag)

    return ImageAnalysis(
        width=w,
        height=h,
        orientation=orientation,
        mean_brightness=mean_brightness,
        mean_saturation=mean_saturation,
        contrast=contrast,
        sharpness=sharpness,
        dom_hue_deg=dom_hue_deg,
        dom_hue_name=dom_name,
        dom_temp=dom_temp,
    )

# -------------------------------
# Inference / Heuristics
# -------------------------------

def time_of_day_suggestion(ana: Optional[ImageAnalysis]) -> str:
    if not ana:
        return "golden hour"  # versatile default
    bright = ana.mean_brightness
    temp = ana.dom_temp or "neutral"
    if bright < 0.35 and temp == "cool":
        return "blue hour / moonlit night"
    if bright > 0.65 and temp == "warm":
        return "golden hour"
    if bright > 0.6:
        return "bright midday"
    if bright < 0.3:
        return "dim interior"
    return "overcast afternoon"

def lighting_setup(ana: Optional[ImageAnalysis]) -> Dict[str, Any]:
    tod = time_of_day_suggestion(ana)
    if not ana:
        key_intensity = "medium"
        fill_ratio = "1:2"
        color_note = "neutral grade"
    else:
        key_intensity = bucketize(ana.mean_brightness, (0.35, 0.65))
        fill_ratio = "1:3" if key_intensity == "low" else ("1:2" if key_intensity == "medium" else "1:1.5")
        if (ana.dom_temp or "neutral") == "warm":
            color_note = "warm tone (subtle amber)"
        elif (ana.dom_temp or "neutral") == "cool":
            color_note = "cool tone (soft blue/cyan)"
        else:
            color_note = "neutral grade"
    return {
        "time_of_day": tod,
        "key_light": f"soft key, {key_intensity} intensity",
        "fill": f"bounce fill, ratio {fill_ratio}",
        "rim": "gentle rim/hair light for separation",
        "practicals": "motivated if present (lamps, signs)",
        "color_grade": color_note + ", gentle contrast curve",
    }

def camera_plan(ana: Optional[ImageAnalysis], seconds: int) -> Dict[str, Any]:
    if ana:
        is_portrait = ana.orientation == "portrait"
        sharp = ana.sharpness
    else:
        is_portrait = False
        sharp = 0.5

    # Movement choice
    if sharp < 0.15:
        movement = "slow tripod push-in"
        stabilization = "locked-off / slider"
    elif sharp < 0.35:
        movement = "gentle gimbal dolly"
        stabilization = "gimbal"
    else:
        movement = "subtle arc left-to-right"
        stabilization = "steady handheld"

    lens = "50mm" if is_portrait else "35mm"
    framing = "medium close-up" if is_portrait else "medium wide"
    fps = 24
    shutter = "180° (≈1/48)"
    ar = "16:9" if not is_portrait else "9:16"

    # Beat counts by duration
    beats = 2 if seconds <= 3 else (3 if seconds <= 5 else 4)
    return {
        "movement": movement,
        "stabilization": stabilization,
        "lens": lens,
        "framing": framing,
        "fps": fps,
        "shutter": shutter,
        "aspect_ratio": ar,
        "beats": beats,
    }

def parse_subject(prompt: str) -> str:
    # Keep neutral & safe; extract a concise subject phrase.
    # Try common "a|an|the <noun>" pattern as a hint
    m = re.search(r"\b(a|an|the)\s+([a-zA-Z0-9\-\s]{2,40})", prompt, flags=re.IGNORECASE)
    if m:
        phrase = m.group(0)
        return phrase.strip()
    # Fallback: first 6 words
    words = prompt.strip().split()
    return " ".join(words[:6]) if words else "primary subject"

def motion_verb_from_prompt(prompt: str) -> str:
    verbs = ["walking", "standing", "running", "turning", "glancing", "breathing", "blinking",
             "floating", "approaching", "looking around", "smiling", "focusing"]
    for v in verbs:
        if re.search(rf"\b{re.escape(v)}\b", prompt, flags=re.IGNORECASE):
            return v
    return "subtly moving"

def style_adjectives(prompt: str, ana: Optional[ImageAnalysis]) -> List[str]:
    out: List[str] = []
    if ana and ana.dom_temp == "warm":
        out.append("warm cinematic")
    elif ana and ana.dom_temp == "cool":
        out.append("cool atmospheric")
    else:
        out.append("neutral cinematic")
    if "film" in prompt.lower():
        out.append("film-like grain")
    if "neon" in prompt.lower():
        out.append("neon glow")
    if ana and ana.contrast > 0.18:
        out.append("rich contrast")
    else:
        out.append("soft contrast")
    return list(dict.fromkeys(out))  # dedupe, preserve order

def actor_notes(prompt: str) -> Dict[str, Any]:
    # Keep generic; avoid sensitive attributes. Treat as "lead actor / subject".
    mood = "focused" if re.search(r"\bfocus|serious|concentrat", prompt.lower()) else "calm"
    wardrobe = "practical, story-appropriate"
    return {
        "role": "lead subject (gender-neutral)",
        "emotion": mood,
        "blocking": "naturalistic micro-movements; hold eyeline on primary point of interest",
        "wardrobe": wardrobe,
    }

def negative_prompts() -> List[str]:
    return [
        "text artifacts",
        "watermarks",
        "deformed anatomy",
        "extra limbs",
        "overexposure, underexposure",
        "flicker, jitter, stutter",
        "unintended zooms",
        "compression artifacts"
    ]

def build_beats(seconds: int, camera_move: str) -> List[Dict[str, Any]]:
    # Time-coded scenario beats
    total = float(seconds)
    if seconds <= 3:
        segments = [0.0, 1.2, total]
        descriptions = [
            "establish framing; hold for read",
            f"{camera_move}; micro action on subject",
        ]
    elif seconds <= 5:
        segments = [0.0, 1.6, 3.6, total]
        descriptions = [
            "establish environment + subject",
            f"{camera_move}; action emphasis",
            "tighten for reaction / detail",
        ]
    else:
        segments = [0.0, 1.6, 3.6, 5.4, total]
        descriptions = [
            "establish environment + subject",
            f"{camera_move}; escalate motion subtly",
            "insert cutaway or detail for texture",
            "return to subject; resolve"
        ]

    beats = []
    for i in range(len(descriptions)):
        beats.append({
            "start": round(segments[i], 2),
            "end": round(segments[i+1], 2),
            "action": descriptions[i],
        })
    return beats

def assemble_wan22(subject: str, scene: str, motion: str, camera_work: str, visual: str) -> str:
    # WAN 2.2 canonical ordering: Subject + Scene + Motion + Camera Work + Visual Style/Lighting
    return (f"[Subject] {subject}; "
            f"[Scene] {scene}; "
            f"[Motion] {motion}; "
            f"[Camera Work] {camera_work}; "
            f"[Visual Style/Lighting] {visual}")

def generate(prompt: str,
             image_path: Optional[str],
             seconds: int,
             seed: Optional[int],
             fps: int,
             aspect_ratio: Optional[str]) -> Dict[str, Any]:

    rng = np.random.default_rng(seed if seed is not None else None)

    ana: Optional[ImageAnalysis] = None
    if image_path:
        im = safe_open_image(image_path)
        if im:
            ana = analyze_image(im)

    # Core pieces
    subj = parse_subject(prompt)
    mov = motion_verb_from_prompt(prompt)
    cam = camera_plan(ana, seconds)
    light = lighting_setup(ana)
    style = style_adjectives(prompt, ana)
    actor = actor_notes(prompt)

    # Scene summary
    scene_bits = []
    if ana:
        scene_bits.append(f"{ana.orientation} orientation")
        if ana.dom_hue_name:
            scene_bits.append(f"dominant {ana.dom_hue_name} hues")
    scene_bits.append(light["time_of_day"])
    scene = ", ".join(scene_bits)

    # Camera work full string
    camera_work = (f"{cam['movement']} with {cam['stabilization']}, "
                   f"{cam['framing']} on {cam['lens']} lens, {cam['fps']}fps, "
                   f"{cam['shutter']}, AR {aspect_ratio or cam['aspect_ratio']}")

    visual = ", ".join(style + [light["color_grade"]])

    wan22 = assemble_wan22(
        subject=subj,
        scene=scene,
        motion=f"{mov}; natural micro-movements; avoid jitter",
        camera_work=camera_work,
        visual=visual
    )

    # Build scenario beats
    beats = build_beats(seconds, cam["movement"])

    # Technicals (allow overrides)
    tech = {
        "fps": fps if fps else cam["fps"],
        "aspect_ratio": aspect_ratio or cam["aspect_ratio"],
        "lens": cam["lens"],
        "framing": cam["framing"],
        "movement": cam["movement"],
        "shutter": cam["shutter"],
        "stabilization": cam["stabilization"],
        "color_grade": light["color_grade"],
        "exposure_style": bucketize((ana.mean_brightness if ana else 0.55), (0.35, 0.65)),
        "contrast_style": bucketize((ana.contrast if ana else 0.15), (0.12, 0.22)),
    }

    # Output
    return {
        "input": {
            "prompt": prompt,
            "image": image_path,
            "seconds": seconds,
            "seed": seed,
        },
        "analysis": (asdict(ana) if ana else None),
        "enhanced_prompt": wan22,
        "scenario": {
            "beats": beats,
            "total_seconds": seconds
        },
        "camera": {
            "plan": cam,
            "notes": f"Favor motivated movement; keep horizon level; avoid snap reframes in {seconds}s spot."
        },
        "actor": actor,
        "lighting": light,
        "tech": tech,
        "negatives": negative_prompts(),
        "provider_hints": {
            "structure": "WAN 2.2",
            "notes": "If your renderer supports 'image-to-video', use the provided image as initial conditioning.",
        }
    }

def format_text(out: Dict[str, Any]) -> str:
    s = []
    s.append("=== ENHANCED PROMPT (WAN 2.2) ===")
    s.append(out["enhanced_prompt"])
    s.append("")
    s.append("=== SCENARIO (time-coded beats) ===")
    for b in out["scenario"]["beats"]:
        s.append(f"{b['start']:>4.2f}–{b['end']:>4.2f}s  • {b['action']}")
    s.append("")
    s.append("=== CAMERA ===")
    cam = out["camera"]["plan"]
    s.append(f"movement: {cam['movement']} | stabilization: {cam['stabilization']}")
    s.append(f"lens: {cam['lens']} | framing: {cam['framing']} | fps: {cam['fps']} | shutter: {cam['shutter']} | AR: {out['tech']['aspect_ratio']}")
    s.append("")
    s.append("=== ACTOR ===")
    a = out["actor"]
    s.append(f"role: {a['role']} | emotion: {a['emotion']}")
    s.append(f"blocking: {a['blocking']} | wardrobe: {a['wardrobe']}")
    s.append("")
    s.append("=== LIGHTING ===")
    lg = out["lighting"]
    s.append(f"time of day: {lg['time_of_day']}")
    s.append(f"key: {lg['key_light']} | fill: {lg['fill']} | rim: {lg['rim']} | practicals: {lg['practicals']}")
    s.append(f"grade: {lg['color_grade']}")
    s.append("")
    s.append("=== TECH ===")
    t = out["tech"]
    s.append(f"fps: {t['fps']} | AR: {t['aspect_ratio']} | lens: {t['lens']} | framing: {t['framing']}")
    s.append(f"movement: {t['movement']} | shutter: {t['shutter']} | stabilization: {t['stabilization']}")
    s.append(f"exposure: {t['exposure_style']} | contrast: {t['contrast_style']}")
    s.append("")
    s.append("=== NEGATIVES ===")
    s.append(", ".join(out["negatives"]))
    return "\n".join(s)

def main():
    p = argparse.ArgumentParser(description="Single-file WAN 2.2 prompt enhancer for short videos.")
    p.add_argument("--prompt", required=True, help="User prompt / idea")
    p.add_argument("--image", default=None, help="Optional reference image path")
    p.add_argument("--seconds", type=int, default=5, choices=[3,5,7], help="Requested video length")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic choices")
    p.add_argument("--fps", type=int, default=24, help="Output fps (tech hint)")
    p.add_argument("--ar", dest="aspect_ratio", default=None, help="Aspect ratio override, e.g., 16:9 or 9:16")
    p.add_argument("--format", choices=["json","text"], default="json", help="Output format")
    args = p.parse_args()

    out = generate(
        prompt=args.prompt,
        image_path=args.image,
        seconds=args.seconds,
        seed=args.seed,
        fps=args.fps,
        aspect_ratio=args.aspect_ratio
    )

    if args.format == "json":
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_text(out))

if __name__ == "__main__":
    main()
