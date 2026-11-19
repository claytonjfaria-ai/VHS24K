"""Standalone GPU frame-restoration & AI Quality Review worker for VHS24K."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from vhs24k.encoding import (
    CODEC_CHOICES,
    build_chunk_writer_cmd,
    build_encoder_spec,
    build_legacy_chunk_writer_cmd,
    encoder_works,
    resolve_encoder,
)
from vhs24k.frame_pipeline import FfmpegSink, run_pipeline

logger = logging.getLogger(__name__)

TEMPORAL_SMOOTH_STRENGTH = 0.48
CALIBRATION_FRAME_INDEX = 15


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for ``vhs24k.enhancer_worker``."""
    parser = argparse.ArgumentParser(
        prog="python -m vhs24k.enhancer_worker",
        description="Upscale a single video chunk with Real-ESRGAN, GFPGAN, and AI Quality Review.",
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input chunk .mp4 path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output upscaled .mp4 path or directory.",
    )
    parser.add_argument(
        "-n",
        "--model-name",
        default="realesr-general-x4v3",
        help="Super-resolution model name (default: realesr-general-x4v3).",
    )
    parser.add_argument(
        "--realesrgan-dir",
        type=Path,
        default=Path("/kaggle/working/Real-ESRGAN"),
        help="Path to cloned xinntao/Real-ESRGAN repository.",
    )
    parser.add_argument("--tile", type=int, default=512, help="Tile size for large frames (default: 512).")
    parser.add_argument(
        "--no-tile-max",
        type=int,
        default=1280,
        help="Frames whose longest side is at most this are processed without tiling (default: 1280).",
    )
    parser.add_argument("--dn", type=float, default=0.65, help="Denoise strength for general-v3 (default: 0.65).")
    parser.add_argument("--face-enhance", action="store_true", help="Enable GFPGAN face restoration.")
    parser.add_argument(
        "--face-weight",
        type=float,
        default=0.85,
        help="GFPGAN blending weight in [0.0, 1.0] (default: 0.85).",
    )
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16.")
    parser.add_argument(
        "--quality-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable closed-loop AI Quality Review & 4K camera polish (default: True).",
    )
    parser.add_argument(
        "--gpu-post",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the 4K polish and temporal smoothing on the GPU (default: True).",
    )
    parser.add_argument(
        "--target-quality-score",
        type=float,
        default=82.0,
        help="Target AI quality review score in [0, 100] (default: 82.0).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional directory for quality report JSON and before/after 4K comparison JPG.",
    )
    parser.add_argument(
        "--codec",
        choices=CODEC_CHOICES,
        default=None,
        help="Encode the chunk to its final 4K format with this codec. "
        "Omit for the legacy unscaled intermediate.",
    )
    parser.add_argument("--crf", type=int, default=17, help="Encoder quality (CRF/CQ base, default: 17).")
    parser.add_argument("--preset", default="auto", help="Encoder preset (default: auto per codec).")
    parser.add_argument(
        "--scale-filter",
        default=None,
        help="FFmpeg scale filter chain applied before encoding (e.g. the 4K scale/pad filter).",
    )
    return parser


def _resolve_output_file(input_path: Path, output_arg: Path) -> Path:
    """Resolve ``-o`` to a concrete ``.mp4`` file path."""
    if output_arg.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}:
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        return output_arg
    output_arg.mkdir(parents=True, exist_ok=True)
    return output_arg / f"{input_path.stem}_out{input_path.suffix}"


def _average_metrics(metrics_list: list[Any]) -> Any:
    """Average numeric fields and union deficiencies across ``FrameQualityMetrics``."""
    if len(metrics_list) == 1:
        return metrics_list[0]
    cls = type(metrics_list[0])
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        vals = [getattr(m, field.name) for m in metrics_list]
        if field.name == "deficiencies":
            seen: list[str] = []
            for v_list in vals:
                for item in v_list:
                    if item not in seen:
                        seen.append(item)
            kwargs[field.name] = seen
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            avg = sum(vals) / len(vals)
            kwargs[field.name] = int(round(avg)) if isinstance(vals[0], int) else round(float(avg), 2)
        else:
            kwargs[field.name] = vals[0]
    return cls(**kwargs)


def _sample_chunk_metrics(
    input_path: Path,
    total_frames: int,
    base_face_weight: float,
) -> tuple[Any, tuple[int, int, int, int]]:
    """Sample up to 5 evenly-spaced frames, lock overscan crop, and derive ``EnhancementRecipe``."""
    import cv2  # type: ignore[import-not-found]
    from vhs24k.quality_review import (
        analyze_frame,
        apply_overscan_crop,
        compute_safe_overscan_crop,
        derive_enhancement_recipe,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for quality sampling: {input_path}")

    try:
        if total_frames > 1:
            num_samples = min(5, total_frames)
            indices = sorted({int(round(i * (total_frames - 1) / max(1, num_samples - 1))) for i in range(num_samples)})
        else:
            indices = [0]

        sampled_frames: list[Any] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                sampled_frames.append(frame)

        if not sampled_frames:
            raise RuntimeError(f"Failed to sample any frames from {input_path}")

        crop_box = compute_safe_overscan_crop(sampled_frames)
        metrics_list = [
            analyze_frame(apply_overscan_crop(frame, crop_box))
            for frame in sampled_frames
        ]
        avg_metrics = _average_metrics(metrics_list)
        recipe = derive_enhancement_recipe(avg_metrics, base_face_weight=base_face_weight)
        return recipe, crop_box
    finally:
        cap.release()


def _ensure_weight(url: str, dest: Path) -> None:
    """Download a model weight file once."""
    if dest.exists():
        return
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https release URLs


def _init_upsampler(args: argparse.Namespace, tile: int) -> tuple[Any, Any | None]:
    """Initialize ``RealESRGANer`` and optional ``GFPGANer`` on GPU/CPU."""
    realesrgan_dir = Path(args.realesrgan_dir)
    local_pkg_dir = realesrgan_dir / "realesrgan"
    local_version_py = local_pkg_dir / "version.py"
    if local_pkg_dir.is_dir() and not local_version_py.exists():
        local_version_py.write_text(
            "__version__ = '0.3.0'\n__gitsha__ = 'unknown'\nversion_info = (0, 3, 0)\n",
            encoding="utf-8",
        )

    import torch  # type: ignore[import-not-found]
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore[import-not-found]
    from realesrgan import RealESRGANer  # type: ignore[import-not-found]
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore[import-not-found]

    # Frame and face-crop shapes are constant within a chunk, so autotuning pays off.
    torch.backends.cudnn.benchmark = True
    use_half = (not args.fp32) and torch.cuda.is_available()
    release = "https://github.com/xinntao/Real-ESRGAN/releases/download"

    if args.model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        local_weight = realesrgan_dir / "weights" / "RealESRGAN_x4plus.pth"
        _ensure_weight(f"{release}/v0.1.0/RealESRGAN_x4plus.pth", local_weight)
        model_path: str | list[str] = str(local_weight)
        dni_weight = None
    elif args.model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
        )
        weights_dir = realesrgan_dir / "weights"
        main_w = weights_dir / "realesr-general-x4v3.pth"
        wdn_w = weights_dir / "realesr-general-wdn-x4v3.pth"
        _ensure_weight(f"{release}/v0.2.5.0/realesr-general-x4v3.pth", main_w)
        _ensure_weight(f"{release}/v0.2.5.0/realesr-general-wdn-x4v3.pth", wdn_w)
        if args.dn <= 0.0:
            model_path, dni_weight = str(main_w), None
        elif args.dn >= 1.0:
            model_path, dni_weight = str(wdn_w), None
        else:
            model_path = [str(main_w), str(wdn_w)]
            dni_weight = [1.0 - args.dn, args.dn]
    else:
        raise ValueError(f"Unsupported model_name for enhancer_worker: {args.model_name!r}")

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
    )

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer  # type: ignore[import-not-found]

        gfpgan_local = realesrgan_dir / "gfpgan" / "weights" / "GFPGANv1.4.pth"
        gfpgan_path = (
            str(gfpgan_local)
            if gfpgan_local.exists()
            else "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        )
        face_enhancer = GFPGANer(
            model_path=gfpgan_path,
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    return upsampler, face_enhancer


def _effective_tile(args: argparse.Namespace, crop_w: int, crop_h: int) -> int:
    """SD frames fit in GPU memory whole; skipping tiling avoids seams and per-tile overhead."""
    return 0 if max(crop_w, crop_h) <= args.no_tile_max else args.tile


def _make_enhance_fn(
    upsampler: Any, face_enhancer: Any | None, face_weight_fn: Callable[[], float]
) -> Callable[[Any], Any]:
    """Return ``pre_frame -> upscaled uint8 BGR`` using GFPGAN or plain Real-ESRGAN."""
    if face_enhancer is not None:

        def enhance_with_faces(frame: Any) -> Any:
            _, _, restored = face_enhancer.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=face_weight_fn(),
            )
            return restored

        return enhance_with_faces

    def enhance_plain(frame: Any) -> Any:
        upscaled, _ = upsampler.enhance(frame, outscale=4)
        return upscaled

    return enhance_plain


def _iter_prepared_frames(
    input_path: Path,
    crop_box: tuple[int, int, int, int],
    recipe: Any | None,
    stats: dict[str, int],
) -> Iterator[Any]:
    """Yield pre-upscale frames (crop, dropout removal, colour fix) in order.

    Runs on a background thread. ``recipe`` is ``None`` when quality review is off,
    in which case frames are yielded untouched.
    """
    import cv2  # type: ignore[import-not-found]
    from vhs24k.quality_review import (
        apply_overscan_crop,
        apply_pre_upscale_color_and_lighting,
        remove_vhs_white_line_dropouts,
    )

    cap = cv2.VideoCapture(str(input_path))
    try:

        def read_cropped() -> Any | None:
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            return apply_overscan_crop(frame, crop_box) if recipe is not None else frame

        prev_raw: Any | None = None
        curr_raw = read_cropped()
        next_raw = read_cropped() if curr_raw is not None else None
        while curr_raw is not None:
            if recipe is not None:
                clean, n_dropouts = remove_vhs_white_line_dropouts(curr_raw, prev_raw, next_raw)
                stats["dropouts"] += n_dropouts
                yield apply_pre_upscale_color_and_lighting(clean, recipe)
            else:
                yield curr_raw
            prev_raw, curr_raw = curr_raw, next_raw
            next_raw = read_cropped() if curr_raw is not None else None
    finally:
        cap.release()


def _calibrate_recipe(
    args: argparse.Namespace,
    input_path: Path,
    crop_box: tuple[int, int, int, int],
    recipe: Any,
    total_frames: int,
    enhance: Callable[[Any], Any],
    post: Any,
    report_dir: Path,
    frame_size: tuple[int, int],
) -> Any:
    """Run the closed-loop quality review once on a representative frame, *before* the chunk.

    The refined recipe is then used for every frame of the chunk, so the whole chunk
    has one consistent look (previously frames before the review used a weaker recipe
    than frames after it).
    """
    import cv2  # type: ignore[import-not-found]
    from vhs24k.quality_review import (
        apply_overscan_crop,
        apply_pre_upscale_color_and_lighting,
        closed_loop_review_and_refine,
        remove_vhs_white_line_dropouts,
        save_comparison_and_report,
    )

    rep_idx = min(CALIBRATION_FRAME_INDEX, max(0, total_frames // 2)) if total_frames > 0 else 0
    cap = cv2.VideoCapture(str(input_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, rep_idx - 1))
        frames: list[Any] = []
        for _ in range(3 if rep_idx > 0 else 2):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(apply_overscan_crop(frame, crop_box))
    finally:
        cap.release()

    offset = 1 if rep_idx > 0 else 0
    if len(frames) <= offset:
        logger.warning("Could not read calibration frame from %s; using the initial recipe.", input_path)
        return recipe
    prev_f = frames[offset - 1] if offset > 0 else None
    curr_f = frames[offset]
    next_f = frames[offset + 1] if len(frames) > offset + 1 else None

    clean, _ = remove_vhs_white_line_dropouts(curr_f, prev_f, next_f)
    upscaled = enhance(apply_pre_upscale_color_and_lighting(clean, recipe))
    initial = post.polish_to_numpy(upscaled, recipe)
    refined, before_m, after_m, final_recipe, was_refined = closed_loop_review_and_refine(
        curr_f,
        initial,
        recipe,
        target_score=args.target_quality_score,
        polish_fn=post.polish_to_numpy,
        raw_upscaled_bgr=upscaled,
    )
    saved = save_comparison_and_report(
        curr_f, refined, before_m, after_m, final_recipe, report_dir, input_path.stem
    )
    width, height = frame_size
    crop_h, crop_w = curr_f.shape[:2]
    print(
        f"[Quality Review] {input_path.stem}: "
        f"Trimmed overscan: {width}x{height} -> {crop_w}x{crop_h} | "
        f"Before={before_m.overall_score:.1f}/100 -> "
        f"After={after_m.overall_score:.1f}/100 "
        f"({after_m.grade}) | "
        f"Auto-Refined={was_refined} | "
        f"Report={saved.get('report_json')}",
        flush=True,
    )
    return final_recipe


def _open_sink(
    args: argparse.Namespace, output_file: Path, out_w: int, out_h: int, fps: float
) -> FfmpegSink:
    """Open the FFmpeg writer: final-quality 4K when ``--codec`` is set, legacy intermediate otherwise."""
    if args.codec:
        if args.codec == "auto":
            encoder = resolve_encoder("auto", args.crf, args.preset)
        else:
            # The parent pinned this codec for every chunk of the source. Switching silently
            # would make the stream-copy join produce a broken file, so fail the chunk instead.
            encoder = build_encoder_spec(args.codec, args.crf, args.preset)
            if not encoder_works(encoder.codec, encoder.args):
                raise RuntimeError(
                    f"Pinned video encoder {args.codec!r} is not usable on this machine; "
                    "refusing to switch codecs mid-source. Re-run with --codec libx265, "
                    "or clear 'final_encode' in the manifest and the finished chunks to restart."
                )
        cmd = build_chunk_writer_cmd(output_file, out_w, out_h, fps, encoder, args.scale_filter)
    else:
        cmd = build_legacy_chunk_writer_cmd(output_file, out_w, out_h, fps)
    return FfmpegSink(cmd)


def process_chunk(args: argparse.Namespace) -> Path:
    """Process a single video chunk end-to-end and return the output file path."""
    import cv2  # type: ignore[import-not-found]

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input chunk not found: {input_path}")

    output_file = _resolve_output_file(input_path, Path(args.output))
    report_dir = Path(args.report_dir) if args.report_dir is not None else output_file.parent

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input chunk: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions ({width}x{height}) for {input_path}")

    recipe = None
    crop_box: tuple[int, int, int, int] = (0, height, 0, width)
    if args.quality_review:
        recipe, crop_box = _sample_chunk_metrics(input_path, total_frames, args.face_weight)

    y0, y1, x0, x1 = crop_box
    crop_w, crop_h = x1 - x0, y1 - y0
    upsampler, face_enhancer = _init_upsampler(args, _effective_tile(args, crop_w, crop_h))

    from vhs24k.gpu_post import make_post_processor

    post = make_post_processor(bool(args.gpu_post) and recipe is not None)
    state: dict[str, Any] = {"recipe": recipe, "prev": None}

    def face_weight() -> float:
        current = state["recipe"]
        return float(getattr(current, "face_weight", args.face_weight)) if current is not None else args.face_weight

    enhance = _make_enhance_fn(upsampler, face_enhancer, face_weight)
    if recipe is not None:
        state["recipe"] = _calibrate_recipe(
            args, input_path, crop_box, recipe, total_frames, enhance, post, report_dir, (width, height)
        )

    def enhance_and_polish(pre_frame: Any) -> Any:
        upscaled = enhance(pre_frame)
        if state["recipe"] is None:
            return upscaled
        polished = post.polish(upscaled, state["recipe"])
        smoothed = post.temporal(polished, state["prev"], TEMPORAL_SMOOTH_STRENGTH)
        state["prev"] = smoothed
        return smoothed

    def to_bytes(frame: Any) -> bytes:
        return post.to_numpy(frame).tobytes() if state["recipe"] is not None else frame.tobytes()

    stats = {"dropouts": 0}
    sink = _open_sink(args, output_file, crop_w * 4, crop_h * 4, fps)
    frames_done = run_pipeline(
        _iter_prepared_frames(input_path, crop_box, state["recipe"], stats),
        enhance_and_polish,
        to_bytes,
        sink,
    )

    logger.info(
        "Processed %d frames (Dropouts fixed=%d px) -> %s",
        frames_done,
        stats["dropouts"],
        output_file,
    )
    return output_file


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``vhs24k.enhancer_worker``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    process_chunk(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
