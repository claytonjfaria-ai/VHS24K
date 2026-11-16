"""Command-line interface for the VHS24K restoration and 4K upscaling pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vhs24k.config import PipelineConfig
from vhs24k.drive_io import download_wedding_files
from vhs24k.postprocess import assemble_final
from vhs24k.preprocess import discover_sources, prepare_chunks, probe_video
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
        "--model",
        default="realesr-general-x4v3",
        help="Real-ESRGAN model name (default: realesr-general-x4v3).",
    )
    parser.add_argument(
        "--no-face-enhance",
        action="store_true",
        help="Disable GFPGAN face enhancement.",
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
        help="Process only the first N seconds as a quick quality test.",
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
        choices=("pillarbox_uhd", "native_4x"),
        default="pillarbox_uhd",
        help="4K output framing mode (default: pillarbox_uhd).",
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
    return PipelineConfig(
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        source_mode=args.source_mode,
        model_name=args.model,
        face_enhance=not args.no_face_enhance,
        chunk_seconds=args.chunk_seconds,
        target_4k_mode=args.target_4k_mode,
        num_gpus=args.num_gpus,
        test_clip_seconds=args.test_clip,
        realesrgan_dir=args.realesrgan_dir,
        dry_run=args.dry_run,
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


def _cmd_prepare(cfg: PipelineConfig) -> int:
    sources = discover_sources(cfg.input_dir, cfg.source_mode)
    for src in sources:
        manifest_path = prepare_chunks(src, cfg.work_dir, cfg)
        print(f"Prepared: {manifest_path}")
    return 0


def _cmd_upscale(cfg: PipelineConfig) -> int:
    for manifest_path in _find_manifests(cfg):
        upscale_manifest(manifest_path, cfg)
        print(f"Upscaled: {manifest_path}")
    return 0


def _cmd_assemble(cfg: PipelineConfig) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for manifest_path in _find_manifests(cfg):
        suffix = "_test_4k.mp4" if cfg.test_clip_seconds is not None else "_4k.mp4"
        out_path = cfg.output_dir / f"{manifest_path.parent.name}{suffix}"
        result = assemble_final(manifest_path, out_path, cfg)
        print(f"Assembled: {result}")
    return 0


def _cmd_run(cfg: PipelineConfig) -> int:
    sources = discover_sources(cfg.input_dir, cfg.source_mode)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        manifest_path = prepare_chunks(src, cfg.work_dir, cfg)
        upscale_manifest(manifest_path, cfg)
        suffix = "_test_4k.mp4" if cfg.test_clip_seconds is not None else "_4k.mp4"
        out_path = cfg.output_dir / f"{src.name}{suffix}"
        result = assemble_final(manifest_path, out_path, cfg)
        print(f"Completed {src.name}: {result}")
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
        "download": lambda: _cmd_download(args, cfg),
    }
    return handlers[args.command]()


if __name__ == "__main__":
    sys.exit(main())
