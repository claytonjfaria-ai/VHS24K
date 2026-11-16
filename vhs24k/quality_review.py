"""Automated AI Quality Review & Closed-Loop Frame Enhancement for VHS24K."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None  # type: ignore

_FACE_CASCADE: Any = None
_FACE_CASCADE_CHECKED = False


@dataclass
class FrameQualityMetrics:
    """Perceptual quality metrics for a single video frame (0-100 scale)."""

    sharpness: float
    face_clarity: float
    dynamic_range: float
    color_naturalness: float
    vhs_artifact_free: float
    overall_score: float
    grade: str
    deficiencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class EnhancementRecipe:
    """Tuned pre-upscale and post-upscale restoration parameters."""

    wb_strength: float = 0.65
    clahe_clip_limit: float = 1.8
    vibrance_boost: float = 0.25
    chroma_denoise: float = 0.50
    micro_contrast: float = 0.45
    face_weight: float = 0.80
    second_pass_sharpen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize recipe parameters to a JSON-compatible dictionary."""
        return asdict(self)


def _validate_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Validate that input is a non-empty (H, W, 3) BGR image."""
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(frame_bgr).__name__}")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) BGR frame, got shape {frame_bgr.shape}")
    if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
        raise ValueError(f"Frame has zero area: {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        return np.clip(frame_bgr, 0, 255).astype(np.uint8)
    return frame_bgr


def _bgr_to_luma(bgr: np.ndarray) -> np.ndarray:
    """Compute Rec.709 luma (float32 in [0, 255]) from a BGR image."""
    f = bgr.astype(np.float32)
    return 0.0722 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.2126 * f[:, :, 2]


def _extract_active_region(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop away solid black pillarbox or letterbox bars if present."""
    luma = _bgr_to_luma(frame_bgr)
    col_active = np.where(luma.mean(axis=0) > 8.0)[0]
    row_active = np.where(luma.mean(axis=1) > 8.0)[0]
    if col_active.size < 16 or row_active.size < 16:
        return frame_bgr
    y0, y1 = int(row_active[0]), int(row_active[-1]) + 1
    x0, x1 = int(col_active[0]), int(col_active[-1]) + 1
    if (y1 - y0) * (x1 - x0) < 0.1 * frame_bgr.shape[0] * frame_bgr.shape[1]:
        return frame_bgr
    return frame_bgr[y0:y1, x0:x1]


def _box_blur_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast O(H*W) separable 2D box blur using cumulative sums in pure numpy."""
    if radius <= 0:
        return arr.astype(np.float32, copy=True)
    f = arr.astype(np.float32, copy=False)
    k = 2 * radius + 1

    padded_x = np.pad(f, ((0, 0), (radius + 1, radius)), mode="edge")
    cs_x = np.cumsum(padded_x, axis=1, dtype=np.float64)
    blur_x = ((cs_x[:, k:] - cs_x[:, :-k]) / k).astype(np.float32)

    padded_y = np.pad(blur_x, ((radius + 1, radius), (0, 0)), mode="edge")
    cs_y = np.cumsum(padded_y, axis=0, dtype=np.float64)
    return ((cs_y[k:, :] - cs_y[:-k, :]) / k).astype(np.float32)


def _smooth_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a 2D float32 array using cv2.GaussianBlur or pure-numpy 2-pass box blur."""
    if radius <= 0:
        return arr.astype(np.float32, copy=True)
    if cv2 is not None:
        k = 2 * radius + 1
        return cv2.GaussianBlur(arr.astype(np.float32), (k, k), 0)
    return _box_blur_2d(_box_blur_2d(arr, radius), max(1, radius // 2))


def _bgr_to_ycrcb_np(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 BGR image to float32 (Y, Cr, Cb) channels."""
    f = bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


def _ycrcb_to_bgr_np(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Convert float32 (Y, Cr, Cb) channels back to uint8 BGR."""
    cr_c = cr - 128.0
    cb_c = cb - 128.0
    r = y + 1.403 * cr_c
    g = y - 0.714 * cr_c - 0.344 * cb_c
    b = y + 1.773 * cb_c
    out = np.stack([b, g, r], axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _laplacian_energy(luma: np.ndarray) -> float:
    """Compute 90th-percentile absolute 3x3 Laplacian response on a 2D luma array."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    lap = (
        luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
        - 4.0 * luma[1:-1, 1:-1]
    )
    return float(np.percentile(np.abs(lap), 90))


def _tenengrad_energy(luma: np.ndarray) -> float:
    """Compute Sobel/Tenengrad gradient magnitude energy."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    gx = (
        (luma[:-2, 2:] + 2.0 * luma[1:-1, 2:] + luma[2:, 2:])
        - (luma[:-2, :-2] + 2.0 * luma[1:-1, :-2] + luma[2:, :-2])
    ) / 8.0
    gy = (
        (luma[2:, :-2] + 2.0 * luma[2:, 1:-1] + luma[2:, 2:])
        - (luma[:-2, :-2] + 2.0 * luma[:-2, 1:-1] + luma[:-2, 2:])
    ) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag, 90))


def _compute_sharpness(luma: np.ndarray) -> float:
    """Compute multi-scale sharpness score in [0, 100]."""
    lap_full = _laplacian_energy(luma)
    ten_full = _tenengrad_energy(luma)
    luma_half = luma[::2, ::2]
    lap_half = _laplacian_energy(luma_half) if min(luma_half.shape) >= 4 else lap_full

    energy = 0.45 * lap_full + 0.35 * ten_full + 0.20 * lap_half
    score = 100.0 * (1.0 - float(np.exp(-energy / 16.0)))
    return float(np.clip(score, 0.0, 100.0))


def _get_face_cascade() -> Any:
    """Lazily load OpenCV Haar face cascade if cv2 is available."""
    global _FACE_CASCADE, _FACE_CASCADE_CHECKED
    if _FACE_CASCADE_CHECKED:
        return _FACE_CASCADE
    _FACE_CASCADE_CHECKED = True
    if cv2 is None:
        return None
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.is_file():
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                _FACE_CASCADE = cascade
    except Exception:
        _FACE_CASCADE = None
    return _FACE_CASCADE


def _compute_face_clarity(active_bgr: np.ndarray, luma: np.ndarray) -> float:
    """Compute facial clarity (via Haar cascade if available) or center-weighted texture."""
    cascade = _get_face_cascade()
    if cascade is not None and cv2 is not None:
        gray_u8 = np.clip(luma, 0, 255).astype(np.uint8)
        faces = cascade.detectMultiScale(gray_u8, scaleFactor=1.15, minNeighbors=5, minSize=(32, 32))
        if len(faces) > 0:
            energies = []
            for x, y, w, h in faces:
                patch = luma[y : y + h, x : x + w]
                if patch.shape[0] >= 8 and patch.shape[1] >= 8:
                    energies.append(0.6 * _laplacian_energy(patch) + 0.4 * _tenengrad_energy(patch))
            if energies:
                face_energy = float(np.mean(energies))
                return float(np.clip(100.0 * (1.0 - np.exp(-face_energy / 14.0)), 0.0, 100.0))

    h, w = luma.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    center_weights = np.exp(-2.5 * (xx * xx + yy * yy))
    detail = np.abs(luma - _smooth_2d(luma, 2))
    weighted_detail = float(np.sum(detail * center_weights) / max(float(np.sum(center_weights)), 1e-6))
    center_lap = _laplacian_energy(
        luma[h // 6 : max(h // 6 + 4, 5 * h // 6), w // 6 : max(w // 6 + 4, 5 * w // 6)]
    )
    combined = 0.55 * center_lap + 1.8 * weighted_detail
    return float(np.clip(100.0 * (1.0 - np.exp(-combined / 14.0)), 0.0, 100.0))


def _compute_dynamic_range(luma: np.ndarray) -> float:
    """Compute luma p98-p2 spread, histogram entropy, and clipping penalty in [0, 100]."""
    p2, p98 = np.percentile(luma, [2.0, 98.0])
    spread_norm = min(1.0, float(p98 - p2) / 215.0)

    hist, _ = np.histogram(luma, bins=64, range=(0.0, 255.0))
    probs = hist.astype(np.float64) / max(float(luma.size), 1.0)
    nz = probs[probs > 0]
    entropy = -float(np.sum(nz * np.log2(nz)))
    entropy_norm = min(1.0, entropy / np.log2(64.0))

    crushed = float(np.mean(luma <= 4.0))
    blown = float(np.mean(luma >= 251.0))
    clip_penalty = min(1.0, (max(0.0, crushed - 0.03) + max(0.0, blown - 0.03)) * 4.0)

    score = 100.0 * (0.55 * spread_norm + 0.30 * entropy_norm + 0.15 * (1.0 - clip_penalty))
    return float(np.clip(score, 0.0, 100.0))


def _compute_color_naturalness(active_bgr: np.ndarray, luma: np.ndarray) -> tuple[float, float, float]:
    """Return (color_naturalness_score, colorfulness, cast_ratio)."""
    f = active_bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    # Ideal natural video colorfulness sits around 25-65
    if colorfulness < 25.0:
        color_score = 45.0 + 55.0 * (colorfulness / 25.0)
    elif colorfulness <= 70.0:
        color_score = 100.0
    else:
        color_score = max(55.0, 100.0 - (colorfulness - 70.0) * 0.9)

    mid_mask = (luma > 25.0) & (luma < 230.0)
    if np.any(mid_mask):
        mr = float(np.mean(r[mid_mask]))
        mg = float(np.mean(g[mid_mask]))
        mb = float(np.mean(b[mid_mask]))
    else:
        mr, mg, mb = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))

    avg_gray = max((mr + mg + mb) / 3.0, 1.0)
    cast_ratio = max(abs(mr - mg), abs(mg - mb), abs(mr - mb)) / avg_gray
    neutrality_score = max(0.0, 100.0 * (1.0 - max(0.0, cast_ratio - 0.04) * 2.4))

    combined = 0.50 * color_score + 0.50 * neutrality_score
    return float(np.clip(combined, 0.0, 100.0)), colorfulness, cast_ratio


def _compute_vhs_artifact_free(active_bgr: np.ndarray, luma: np.ndarray) -> float:
    """Measure absence of horizontal interlace combing and chroma noise in [0, 100]."""
    if luma.shape[0] >= 4:
        even = luma[0:-2:2, :]
        odd = luma[1:-1:2, :]
        even_next = luma[2::2, :]
        n_rows = min(even.shape[0], odd.shape[0], even_next.shape[0])
        comb = np.abs(odd[:n_rows] - 0.5 * (even[:n_rows] + even_next[:n_rows]))
        horiz = np.abs(luma[: n_rows * 2, 1:] - luma[: n_rows * 2, :-1])
        comb_ratio = float(np.mean(comb)) / max(float(np.mean(horiz)) + 2.0, 1.0)
        comb_penalty = max(0.0, (comb_ratio - 0.85) * 35.0)
    else:
        comb_penalty = 0.0

    _, cr, cb = _bgr_to_ycrcb_np(active_bgr)
    cr_noise = float(np.std(cr - _smooth_2d(cr, 2)))
    cb_noise = float(np.std(cb - _smooth_2d(cb, 2)))
    chroma_noise = 0.5 * (cr_noise + cb_noise)
    chroma_penalty = max(0.0, (chroma_noise - 1.5) * 6.5)

    score = 100.0 - comb_penalty - chroma_penalty
    return float(np.clip(score, 0.0, 100.0))


def _grade_from_score(score: float) -> str:
    """Map numeric overall score to human-readable quality grade."""
    if score >= 85.0:
        return "A+ (Modern 4K Camera)"
    if score >= 75.0:
        return "A (4K Cinema)"
    if score >= 60.0:
        return "B (Enhanced HD)"
    return "C (Needs Re-Enhancement)"


def analyze_frame(frame_bgr: np.ndarray) -> FrameQualityMetrics:
    """Analyze a uint8 BGR frame and return perceptual quality metrics and deficiencies."""
    frame_bgr = _validate_bgr_frame(frame_bgr)
    active_bgr = _extract_active_region(frame_bgr)
    luma = _bgr_to_luma(active_bgr)

    sharpness = _compute_sharpness(luma)
    face_clarity = _compute_face_clarity(active_bgr, luma)
    dynamic_range = _compute_dynamic_range(luma)
    color_naturalness, colorfulness, cast_ratio = _compute_color_naturalness(active_bgr, luma)
    vhs_artifact_free = _compute_vhs_artifact_free(active_bgr, luma)

    overall_score = float(
        np.clip(
            0.28 * sharpness
            + 0.20 * face_clarity
            + 0.18 * dynamic_range
            + 0.18 * color_naturalness
            + 0.16 * vhs_artifact_free,
            0.0,
            100.0,
        )
    )

    deficiencies: list[str] = []
    if sharpness < 72.0:
        deficiencies.append("soft_scene_textures")
    if face_clarity < 72.0:
        deficiencies.append("soft_faces")
    if dynamic_range < 72.0:
        deficiencies.append("compressed_dynamic_range")
    if cast_ratio > 0.12 or color_naturalness < 70.0:
        deficiencies.append("vhs_color_cast")
    if colorfulness < 22.0:
        deficiencies.append("muted_saturation")
    if vhs_artifact_free < 78.0:
        deficiencies.append("chroma_noise")

    return FrameQualityMetrics(
        sharpness=round(sharpness, 2),
        face_clarity=round(face_clarity, 2),
        dynamic_range=round(dynamic_range, 2),
        color_naturalness=round(color_naturalness, 2),
        vhs_artifact_free=round(vhs_artifact_free, 2),
        overall_score=round(overall_score, 2),
        grade=_grade_from_score(overall_score),
        deficiencies=deficiencies,
    )


def derive_enhancement_recipe(
    metrics: FrameQualityMetrics,
    base_face_weight: float = 0.80,
) -> EnhancementRecipe:
    """Derive a scene-adaptive enhancement recipe from FrameQualityMetrics."""
    defs = set(metrics.deficiencies)

    wb_strength = 0.85 if "vhs_color_cast" in defs else 0.55
    clahe_clip_limit = 2.4 if "compressed_dynamic_range" in defs else 1.6
    vibrance_boost = 0.40 if "muted_saturation" in defs else 0.22
    chroma_denoise = 0.75 if "chroma_noise" in defs else 0.40
    micro_contrast = 0.62 if "soft_scene_textures" in defs else 0.42
    face_weight = min(1.0, max(base_face_weight, 0.85)) if "soft_faces" in defs else base_face_weight

    return EnhancementRecipe(
        wb_strength=round(wb_strength, 3),
        clahe_clip_limit=round(clahe_clip_limit, 3),
        vibrance_boost=round(vibrance_boost, 3),
        chroma_denoise=round(chroma_denoise, 3),
        micro_contrast=round(micro_contrast, 3),
        face_weight=round(face_weight, 3),
        second_pass_sharpen=0.0,
    )


def apply_pre_upscale_color_and_lighting(
    frame_bgr: np.ndarray,
    recipe: EnhancementRecipe,
) -> np.ndarray:
    """Clean chroma noise, neutralize VHS color cast, expand dynamic range, and boost vibrance."""
    frame_bgr = _validate_bgr_frame(frame_bgr)

    # 1. Chroma noise/bleed cleanup in YCrCb (Y luma stays 100% sharp)
    y, cr, cb = _bgr_to_ycrcb_np(frame_bgr)
    if recipe.chroma_denoise > 0.0:
        radius = 2 if recipe.chroma_denoise < 0.65 else 3
        cr_smooth = _smooth_2d(cr, radius)
        cb_smooth = _smooth_2d(cb, radius)
        alpha = float(np.clip(recipe.chroma_denoise, 0.0, 1.0))
        cr = cr * (1.0 - alpha) + cr_smooth * alpha
        cb = cb * (1.0 - alpha) + cb_smooth * alpha
    work = _ycrcb_to_bgr_np(y, cr, cb).astype(np.float32)

    # 2. Highlight-protected Gray-World Auto White Balance
    if recipe.wb_strength > 0.0:
        luma = _bgr_to_luma(work)
        valid = (luma > 15.0) & (luma < 235.0)
        if np.any(valid):
            mean_b = max(float(np.mean(work[:, :, 0][valid])), 1.0)
            mean_g = max(float(np.mean(work[:, :, 1][valid])), 1.0)
            mean_r = max(float(np.mean(work[:, :, 2][valid])), 1.0)
        else:
            mean_b = max(float(np.mean(work[:, :, 0])), 1.0)
            mean_g = max(float(np.mean(work[:, :, 1])), 1.0)
            mean_r = max(float(np.mean(work[:, :, 2])), 1.0)

        gray = (mean_b + mean_g + mean_r) / 3.0
        gains = np.array(
            [
                np.clip(gray / mean_b, 0.70, 1.40),
                np.clip(gray / mean_g, 0.70, 1.40),
                np.clip(gray / mean_r, 0.70, 1.40),
            ],
            dtype=np.float32,
        )
        highlight_protect = np.clip((255.0 - luma) / 45.0, 0.0, 1.0)[:, :, None]
        eff_strength = float(np.clip(recipe.wb_strength, 0.0, 1.0)) * highlight_protect
        balanced = work * gains[None, None, :]
        work = work * (1.0 - eff_strength) + balanced * eff_strength

    work_u8 = np.clip(np.rint(work), 0, 255).astype(np.uint8)

    # 3. Adaptive Dynamic Range expansion (CLAHE in LAB if cv2 present, else numpy S-curve)
    if recipe.clahe_clip_limit > 0.0:
        if cv2 is not None:
            lab = cv2.cvtColor(work_u8, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=float(max(0.5, recipe.clahe_clip_limit)),
                tileGridSize=(8, 8),
            )
            l_eq = clahe.apply(l_chan)
            l_blend = np.clip(0.35 * l_chan.astype(np.float32) + 0.65 * l_eq.astype(np.float32), 0, 255).astype(np.uint8)
            work_u8 = cv2.cvtColor(cv2.merge([l_blend, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
        else:
            y_c, cr_c, cb_c = _bgr_to_ycrcb_np(work_u8)
            p1, p99 = np.percentile(y_c, [1.0, 99.0])
            if p99 - p1 > 10.0:
                y_norm = np.clip((y_c - p1) / (p99 - p1), 0.0, 1.0)
                s_strength = float(np.clip((recipe.clahe_clip_limit - 1.0) * 0.25, 0.05, 0.45))
                y_scurve = y_norm + s_strength * np.sin(2.0 * np.pi * y_norm) * (-0.12)
                y_stretched = np.clip(y_scurve * 255.0, 0.0, 255.0)
                y_out = 0.40 * y_c + 0.60 * y_stretched
                work_u8 = _ycrcb_to_bgr_np(y_out, cr_c, cb_c)

    # 4. Skin-safe Vibrance boost
    if recipe.vibrance_boost > 0.0:
        f = work_u8.astype(np.float32)
        b_ch, g_ch, r_ch = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        sat = (max_c - min_c) / np.maximum(max_c, 1.0)
        skin_mask = (
            (r_ch > g_ch) & (g_ch > b_ch) & ((r_ch - g_ch) > 10.0) & (sat > 0.12) & (sat < 0.65)
        ).astype(np.float32)
        luma_f = _bgr_to_luma(f)
        boost = float(np.clip(recipe.vibrance_boost, 0.0, 1.0)) * (1.0 - sat) * (1.0 - 0.6 * skin_mask)
        f = luma_f[:, :, None] + (f - luma_f[:, :, None]) * (1.0 + boost[:, :, None])
        work_u8 = np.clip(np.rint(f), 0, 255).astype(np.uint8)

    return work_u8


def apply_post_upscale_4k_camera_polish(
    frame_bgr: np.ndarray,
    recipe: EnhancementRecipe,
) -> np.ndarray:
    """Apply luminance-only multi-scale micro-contrast & detail synthesis with halo suppression."""
    frame_bgr = _validate_bgr_frame(frame_bgr)
    y, cr, cb = _bgr_to_ycrcb_np(frame_bgr)

    blur_fine = _smooth_2d(y, 1)
    blur_med = _smooth_2d(y, 4)

    fine_detail = y - blur_fine
    med_detail = blur_fine - blur_med

    g_fine = 0.65 * recipe.micro_contrast + 0.85 * recipe.second_pass_sharpen
    g_med = 0.35 * recipe.micro_contrast + 0.30 * recipe.second_pass_sharpen

    delta = g_fine * fine_detail + g_med * med_detail
    max_delta = 18.0 + 10.0 * recipe.second_pass_sharpen
    # Soft-clip / clamp delta to prevent ringing halos and protect extreme highlights/shadows
    midtone_mask = np.clip(np.minimum(y, 255.0 - y) / 24.0, 0.25, 1.0)
    delta = np.clip(delta, -max_delta, max_delta) * midtone_mask

    y_polished = np.clip(y + delta, 0.0, 255.0)
    return _ycrcb_to_bgr_np(y_polished, cr, cb)


def closed_loop_review_and_refine(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    recipe: EnhancementRecipe,
    target_score: float = 82.0,
) -> tuple[np.ndarray, FrameQualityMetrics, FrameQualityMetrics, EnhancementRecipe, bool]:
    """Evaluate before/after frames and automatically refine if below target_score."""
    before_metrics = analyze_frame(before_bgr)
    after_metrics = analyze_frame(after_bgr)

    if after_metrics.overall_score >= target_score and after_metrics.sharpness >= 75.0:
        return after_bgr, before_metrics, after_metrics, recipe, False

    score_deficit = max(0.0, target_score - after_metrics.overall_score)
    sharp_deficit = max(0.0, 75.0 - after_metrics.sharpness)
    extra_sharpen = float(np.clip(0.35 + 0.015 * max(score_deficit, sharp_deficit), 0.30, 0.75))

    updated_recipe = EnhancementRecipe(
        wb_strength=recipe.wb_strength,
        clahe_clip_limit=recipe.clahe_clip_limit,
        vibrance_boost=recipe.vibrance_boost,
        chroma_denoise=recipe.chroma_denoise,
        micro_contrast=min(0.85, recipe.micro_contrast + 0.15),
        face_weight=recipe.face_weight,
        second_pass_sharpen=round(extra_sharpen, 3),
    )

    refined_bgr = apply_post_upscale_4k_camera_polish(after_bgr, updated_recipe)
    refined_metrics = analyze_frame(refined_bgr)

    if refined_metrics.overall_score >= after_metrics.overall_score:
        return refined_bgr, before_metrics, refined_metrics, updated_recipe, True
    return after_bgr, before_metrics, after_metrics, updated_recipe, True


def _resize_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize a BGR image to target_h preserving aspect ratio."""
    h, w = bgr.shape[:2]
    if h == target_h:
        return bgr
    target_w = max(1, round(w * target_h / h))
    if cv2 is not None:
        return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if Image is not None:
        rgb = bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb).resize((target_w, target_h))
        return np.asarray(pil_img)[:, :, ::-1].copy()

    # Pure-numpy bilinear fallback
    ys = np.linspace(0, h - 1, target_h)
    xs = np.linspace(0, w - 1, target_w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None].astype(np.float32)
    wx = (xs - x0)[None, :, None].astype(np.float32)
    top = (1.0 - wx) * bgr[y0][:, x0] + wx * bgr[y0][:, x1]
    bot = (1.0 - wx) * bgr[y1][:, x0] + wx * bgr[y1][:, x1]
    return np.clip(np.rint((1.0 - wy) * top + wy * bot), 0, 255).astype(np.uint8)


def save_comparison_and_report(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    before_metrics: FrameQualityMetrics,
    after_metrics: FrameQualityMetrics,
    recipe: EnhancementRecipe,
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    """Save JSON quality report and side-by-side before/after 4K comparison JPEG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "before": before_metrics.to_dict(),
        "after": after_metrics.to_dict(),
        "score_delta": round(after_metrics.overall_score - before_metrics.overall_score, 2),
        "grade": after_metrics.grade,
        "fixed_deficiencies": [
            d for d in before_metrics.deficiencies if d not in after_metrics.deficiencies
        ],
        "remaining_deficiencies": list(after_metrics.deficiencies),
        "recipe": recipe.to_dict(),
    }

    report_path = output_dir / f"{prefix}_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    saved: dict[str, str] = {"report_json": str(report_path)}

    if cv2 is not None or Image is not None:
        before_resized = _resize_to_height(before_bgr, after_bgr.shape[0])
        side_by_side = np.concatenate([before_resized, after_bgr], axis=1)
        jpg_path = output_dir / f"{prefix}_before_vs_after_4k.jpg"
        if cv2 is not None:
            cv2.imwrite(str(jpg_path), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 92])
        elif Image is not None:
            Image.fromarray(side_by_side[:, :, ::-1]).save(jpg_path, quality=92)
        saved["comparison_jpg"] = str(jpg_path)

    return saved
