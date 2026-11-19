"""Command-line interface for the VHS24K restoration and 4K upscaling pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from vhs24k.config import PipelineConfig
from vhs24k.drive_io import download_wedding_files
from vhs24k.postprocess import assemble_final
from vhs24k.preprocess import discover_sources, prepare_chunks, probe_video
from vhs24k.progress import RunTracker, describe_work_dir, format_bytes
from vhs24k.upscale import upscale_manifest

logger = logging.getLogger(__name__)


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Register shared pipeline flags on a subcommand parser."""
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing source .VOB or .mp4 files (default: ./input).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work"),
        help="Working directory for chunks and manifests (default: ./work).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for final 4K MP4s (default: ./output).",
    )
    parser.add_argument(
        "--source-mode",
        choices=("auto", "vob", "mp4"),
        default="auto",
        help="Source discovery mode (default: auto).",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional source group name to process (e.g. VTS_01 or VTS_02).",
    )
    parser.add_argument(
        "--model",
        default="realesr-general-x4v3",
        help="Real-ESRGAN model name (default: realesr-general-x4v3).",
    )
    parser.add_argument(
        "--denoise-strength",
        "--dn",
        dest="denoise_strength",
        type=float,
        default=0.65,
        help="Denoise strength in [0.0, 1.0] (default: 0.65).",
    )
    parser.add_argument(
        "--no-face-enhance",
        action="store_true",
        help="Disable GFPGAN face enhancement.",
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=0.85,
        help="GFPGAN face restoration blending weight in [0.0, 1.0] (default: 0.85).",
    )
    parser.add_argument(
        "--no-quality-review",
        action="store_true",
        help="Disable closed-loop AI Quality Review & 4K camera polish.",
    )
    parser.add_argument(
        "--target-quality-score",
        type=float,
        default=82.0,
        help="Target AI quality review score in [0, 100] (default: 82.0).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=60,
        help="Segment duration in seconds for chunked upscaling (default: 60).",
    )
    parser.add_argument(
        "--test-clip",
        type=int,
        default=None,
        help="Process only a short N-second clip as a quick quality test.",
    )
    parser.add_argument(
        "--test-start",
        type=int,
        default=900,
        help="Start offset in seconds when using --test-clip (default: 900s = 15 minutes in, to skip the opening scene and sample the ceremony).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Maximum number of parallel GPU workers (default: 2).",
    )
    parser.add_argument(
        "--4k-mode",
        dest="target_4k_mode",
        choices=("fullscreen_uhd", "pillarbox_uhd", "native_4x"),
        default="fullscreen_uhd",
        help="4K output framing mode (default: fullscreen_uhd).",
    )
    parser.add_argument(
        "--codec",
        dest="video_codec",
        choices=("auto", "hevc_nvenc", "libx265", "libx264"),
        default="auto",
        help="Final 4K video encoder (default: auto = GPU HEVC if usable, else libx265).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=17,
        help="Final video quality; lower is better and larger (default: 17).",
    )
    parser.add_argument(
        "--no-gpu-post",
        action="store_true",
        help="Run the 4K polish/temporal smoothing on the CPU instead of the GPU.",
    )
    parser.add_argument(
        "--realesrgan-dir",
        type=Path,
        default=None,
        help="Path to cloned xinntao/Real-ESRGAN repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/plan commands without executing FFmpeg or Real-ESRGAN.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m vhs24k.cli",
        description="VHS/DVD to 4K restoration and upscaling pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("inspect", "Discover and probe source videos in --input-dir."),
        ("prepare", "Deinterlace, denoise, and split sources into chunks."),
        ("upscale", "Upscale prepared chunks across available GPUs."),
        ("assemble", "Scale upscaled chunks to 4K and mux original audio."),
        ("run", "Run prepare -> upscale -> assemble end-to-end."),
        ("status", "Show percent complete from the work directory (reads files only)."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_flags(sub)

    dl_sub = subparsers.add_parser(
        "download",
        help="Download wedding source files from Google Drive.",
    )
    _add_common_flags(dl_sub)
    dl_sub.add_argument(
        "--method",
        choices=("rclone", "gdown"),
        default="rclone",
        help="Download method (default: rclone).",
    )
    dl_sub.add_argument(
        "--rclone-remote",
        default="gdrive:Wedding",
        help="rclone remote path (default: gdrive:Wedding).",
    )

    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a validated ``PipelineConfig`` from parsed CLI arguments."""
    field_names = {f.name for f in dataclasses.fields(PipelineConfig)}
    extra: dict[str, object] = {}
    if "quality_review" in field_names:
        extra["quality_review"] = not args.no_quality_review
    if "target_quality_score" in field_names:
        extra["target_quality_score"] = args.target_quality_score
    if "source_name" in field_names:
        extra["source_name"] = args.source_name
    extra["video_codec"] = args.video_codec
    extra["crf"] = args.crf
    extra["gpu_post"] = not args.no_gpu_post

    return PipelineConfig(
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        source_mode=args.source_mode,
        denoise_strength=args.denoise_strength,
        model_name=args.model,
        face_enhance=not args.no_face_enhance,
        face_weight=args.face_weight,
        chunk_seconds=args.chunk_seconds,
        target_4k_mode=args.target_4k_mode,
        num_gpus=args.num_gpus,
        test_clip_seconds=args.test_clip,
        test_start_seconds=args.test_start,
        realesrgan_dir=args.realesrgan_dir,
        dry_run=args.dry_run,
        **extra,
    )


def _find_manifests(cfg: PipelineConfig) -> list[Path]:
    """Locate ``manifest.json`` files in ``cfg.work_dir``."""
    manifests = sorted(cfg.work_dir.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No manifest.json files found in {cfg.work_dir}. Run 'prepare' first."
        )
    return manifests


def _cmd_inspect(cfg: PipelineConfig) -> int:
    sources = discover_sources(cfg.input_dir, cfg.source_mode)
    report = []
    for src in sources:
        info = probe_video(src.files[0])
        report.append(
            {
                "name": src.name,
                "source_type": src.source_type,
                "files": [str(p) for p in src.files],
                "first_file_probe": info.to_dict(),
            }
        )
    print(json.dumps(report, indent=2))
    return 0


def _filter_sources(sources: list, cfg: PipelineConfig) -> list:
    """Filter discovered sources by ``cfg.source_name`` when provided."""
    source_name = getattr(cfg, "source_name", None)
    if source_name is not None:
        filtered = [s for s in sources if s.name.lower() == source_name.lower()]
        if not filtered:
            raise ValueError(f"Source group {source_name!r} not found.")
        return filtered
    return sources


def _new_tracker(cfg: PipelineConfig, source_bytes: dict[str, int]) -> RunTracker | None:
    """Progress tracker weighted by source file size (a test clip is weighted by stage only)."""
    if cfg.dry_run:
        return None
    if cfg.test_clip_seconds is not None:
        return RunTracker(cfg.work_dir, {name: 1 for name in source_bytes}, show_source_bytes=False)
    return RunTracker(cfg.work_dir, source_bytes)


def _manifest_source_bytes(manifest_path: Path) -> int:
    """Total size of the original source files recorded in a manifest."""
    try:
        files = json.loads(manifest_path.read_text(encoding="utf-8")).get("source_files", [])
    except (OSError, ValueError, AttributeError):
        return 0
    total = 0
    for name in files:
        try:
            total += Path(name).stat().st_size
        except OSError:
            pass
    return total


def _manifest_tracker(cfg: PipelineConfig, manifests: list[Path]) -> RunTracker | None:
    return _new_tracker(cfg, {m.parent.name: _manifest_source_bytes(m) for m in manifests})


def _cmd_prepare(cfg: PipelineConfig) -> int:
    sources = _filter_sources(discover_sources(cfg.input_dir, cfg.source_mode), cfg)
    tracker = _new_tracker(cfg, {s.name: sum(p.stat().st_size for p in s.files) for s in sources})
    for src in sources:
        if tracker is not None:
            tracker.begin_source(src.name)
        manifest_path = prepare_chunks(src, cfg.work_dir, cfg, tracker)
        if tracker is not None:
            tracker.finish_source("prepare")
        print(f"Prepared: {manifest_path}")
    if tracker is not None:
        tracker.close("Prepare finished")
    return 0


def _cmd_upscale(cfg: PipelineConfig) -> int:
    manifests = _find_manifests(cfg)
    tracker = _manifest_tracker(cfg, manifests)
    for manifest_path in manifests:
        if tracker is not None:
            tracker.begin_source(manifest_path.parent.name)
        upscale_manifest(manifest_path, cfg, tracker)
        if tracker is not None:
            tracker.finish_source("upscale")
        print(f"Upscaled: {manifest_path}")
    if tracker is not None:
        tracker.close("Upscale finished")
    return 0


def _cmd_assemble(cfg: PipelineConfig) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = _find_manifests(cfg)
    tracker = _manifest_tracker(cfg, manifests)
    for manifest_path in manifests:
        suffix = "_test_4k.mp4" if cfg.test_clip_seconds is not None else "_4k.mp4"
        out_path = cfg.output_dir / f"{manifest_path.parent.name}{suffix}"
        if tracker is not None:
            tracker.begin_source(manifest_path.parent.name)
        result = assemble_final(manifest_path, out_path, cfg, tracker)
        if tracker is not None:
            tracker.finish_source("assemble")
        print(f"Assembled: {result}")
    if tracker is not None:
        tracker.close()
    return 0


def _cmd_run(cfg: PipelineConfig) -> int:
    sources = discover_sources(cfg.input_dir, cfg.source_mode)
    if getattr(cfg, "source_name", None) is not None:
        sources = _filter_sources(sources, cfg)
    elif cfg.test_clip_seconds is not None:
        vts02 = [s for s in sources if s.name.lower() == "vts_02"]
        sources = vts02 if vts02 else sources[:1]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = _new_tracker(cfg, {s.name: sum(p.stat().st_size for p in s.files) for s in sources})
    for src in sources:
        if tracker is not None:
            tracker.begin_source(src.name)
        manifest_path = prepare_chunks(src, cfg.work_dir, cfg, tracker)
        upscale_manifest(manifest_path, cfg, tracker)
        suffix = "_test_4k.mp4" if cfg.test_clip_seconds is not None else "_4k.mp4"
        out_path = cfg.output_dir / f"{src.name}{suffix}"
        result = assemble_final(manifest_path, out_path, cfg, tracker)
        if tracker is not None:
            tracker.finish_source()
        size = f" ({format_bytes(result.stat().st_size)})" if result.exists() else ""
        print(f"Completed {src.name}: {result}{size}")
    if tracker is not None:
        tracker.close()
    return 0


def _cmd_status(cfg: PipelineConfig) -> int:
    manifests = sorted(cfg.work_dir.glob("*/manifest.json"))
    if not manifests:
        print(f"No manifest.json files found in {cfg.work_dir}. Nothing has been prepared yet.")
        return 1
    for line in describe_work_dir(cfg.work_dir, manifests):
        print(line)
    return 0


def _cmd_download(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    source_type = "vob" if cfg.source_mode == "auto" else cfg.source_mode
    paths = download_wedding_files(
        dest_dir=cfg.input_dir,
        source_type=source_type,
        method=args.method,
        rclone_remote=args.rclone_remote,
    )
    for path in paths:
        print(f"Downloaded: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)

    handlers = {
        "inspect": lambda: _cmd_inspect(cfg),
        "prepare": lambda: _cmd_prepare(cfg),
        "upscale": lambda: _cmd_upscale(cfg),
        "assemble": lambda: _cmd_assemble(cfg),
        "run": lambda: _cmd_run(cfg),
        "status": lambda: _cmd_status(cfg),
        "download": lambda: _cmd_download(args, cfg),
    }
    return handlers[args.command]()


if __name__ == "__main__":
    sys.exit(main())
