"""Standalone GPU frame-restoration & AI Quality Review worker for VHS24K."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    parser.add_argument("--tile", type=int, default=512, help="Tile size (default: 512).")
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


def _init_upsampler(args: argparse.Namespace) -> tuple[Any, Any | None]:
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

    use_half = (not args.fp32) and torch.cuda.is_available()

    if args.model_name == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        local_weight = realesrgan_dir / "weights" / "RealESRGAN_x4plus.pth"
        if not local_weight.exists():
            import urllib.request

            local_weight.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                local_weight,
            )
        model_path: str | list[str] = str(local_weight)
        dni_weight = None
    elif args.model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        weights_dir = realesrgan_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        main_w = weights_dir / "realesr-general-x4v3.pth"
        wdn_w = weights_dir / "realesr-general-wdn-x4v3.pth"
        if not main_w.exists():
            import urllib.request

            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                main_w,
            )
        if not wdn_w.exists():
            import urllib.request

            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                wdn_w,
            )
        if args.dn <= 0.0:
            model_path = str(main_w)
            dni_weight = None
        elif args.dn >= 1.0:
            model_path = str(wdn_w)
            dni_weight = None
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
        tile=args.tile,
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


def _open_ffmpeg_writer(output_file: Path, width: int, height: int, fps: float) -> subprocess.Popen[bytes]:
    """Open an FFmpeg rawvideo pipe writer encoding at CRF 16."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _unpack_review_result(result: Any, fallback_frame: Any, fallback_recipe: Any) -> tuple[Any, Any, Any]:
    """Normalize return tuple from ``closed_loop_review_and_refine``."""
    if not isinstance(result, tuple):
        return fallback_frame, fallback_recipe, result
    if len(result) == 3:
        return result[0], result[1], result[2]
    if len(result) == 2:
        first, second = result
        if hasattr(first, "shape"):
            return first, fallback_recipe, second
        return fallback_frame, first, second
    return fallback_frame, fallback_recipe, result


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
        from vhs24k.quality_review import (
            apply_overscan_crop,
            apply_post_upscale_4k_camera_polish,
            apply_pre_upscale_color_and_lighting,
            closed_loop_review_and_refine,
            remove_vhs_white_line_dropouts,
            save_comparison_and_report,
            temporal_smooth_4k_frame,
        )

        recipe, crop_box = _sample_chunk_metrics(input_path, total_frames, args.face_weight)

    upsampler, face_enhancer = _init_upsampler(args)

    y0, y1, x0, x1 = crop_box
    crop_w = x1 - x0
    crop_h = y1 - y0
    out_w, out_h = crop_w * 4, crop_h * 4
    ffmpeg_proc = _open_ffmpeg_writer(output_file, out_w, out_h, fps)
    if ffmpeg_proc.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin pipe.")

    cap = cv2.VideoCapture(str(input_path))
    rep_idx = min(15, max(0, total_frames // 2)) if total_frames > 0 else 0
    reviewed = False
    frame_idx = 0
    total_dropouts_fixed = 0
    prev_4k_frame: Any | None = None

    def _read_cropped() -> Any | None:
        ok, frm = cap.read()
        if not ok or frm is None:
            return None
        if args.quality_review and recipe is not None:
            return apply_overscan_crop(frm, crop_box)
        return frm

    try:
        prev_raw: Any | None = None
        curr_raw = _read_cropped()
        next_raw = _read_cropped() if curr_raw is not None else None

        while curr_raw is not None:
            if args.quality_review and recipe is not None:
                clean_raw, n_dropouts = remove_vhs_white_line_dropouts(
                    curr_raw, prev_raw, next_raw
                )
                total_dropouts_fixed += n_dropouts
                pre_frame = apply_pre_upscale_color_and_lighting(clean_raw, recipe)
            else:
                pre_frame = curr_raw

            if face_enhancer is not None:
                weight = float(getattr(recipe, "face_weight", args.face_weight)) if recipe is not None else args.face_weight
                _, _, upscaled_frame = face_enhancer.enhance(
                    pre_frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=weight,
                )
            else:
                upscaled_frame, _ = upsampler.enhance(pre_frame, outscale=4)

            if args.quality_review and recipe is not None:
                post_frame = apply_post_upscale_4k_camera_polish(upscaled_frame, recipe)
                if not reviewed and frame_idx >= rep_idx:
                    (
                        post_frame,
                        before_metrics,
                        after_metrics,
                        recipe,
                        was_refined,
                    ) = closed_loop_review_and_refine(
                        curr_raw,
                        post_frame,
                        recipe,
                        target_score=args.target_quality_score,
                    )
                    saved = save_comparison_and_report(
                        curr_raw,
                        post_frame,
                        before_metrics,
                        after_metrics,
                        recipe,
                        report_dir,
                        input_path.stem,
                    )
                    print(
                        f"[Quality Review] {input_path.stem}: "
                        f"Trimmed overscan: {width}x{height} -> {crop_w}x{crop_h} | "
                        f"Before={before_metrics.overall_score:.1f}/100 -> "
                        f"After={after_metrics.overall_score:.1f}/100 "
                        f"({after_metrics.grade}) | "
                        f"Auto-Refined={was_refined} | "
                        f"Dropouts fixed={total_dropouts_fixed} px | "
                        f"Report={saved.get('report_json')}",
                        flush=True,
                    )
                    reviewed = True
                post_frame = temporal_smooth_4k_frame(post_frame, prev_4k_frame, strength=0.48)
                prev_4k_frame = post_frame
            else:
                post_frame = upscaled_frame

            ffmpeg_proc.stdin.write(post_frame.tobytes())
            prev_raw = curr_raw
            curr_raw = next_raw
            next_raw = _read_cropped() if curr_raw is not None else None
            frame_idx += 1

        ffmpeg_proc.stdin.close()
        stderr_bytes = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr is not None else b""
        rc = ffmpeg_proc.wait()
        if rc != 0:
            err_tail = stderr_bytes.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"FFmpeg writer failed (exit {rc}): {err_tail}")
    finally:
        cap.release()
        if ffmpeg_proc.poll() is None:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait()

    logger.info(
        "Processed %d frames (Dropouts fixed=%d px) -> %s",
        frame_idx,
        total_dropouts_fixed,
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
