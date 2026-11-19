"""Video encoder selection and FFmpeg writer command construction.

The pipeline encodes every upscaled chunk to its *final* 4K format inside the
GPU worker (concurrently with inference), so the last assembly step can simply
stream-copy the chunks instead of re-encoding hours of 4K video on the CPU.
"""

from __future__ import annotations

import functools
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CODEC_CHOICES = ("auto", "hevc_nvenc", "libx265", "libx264")

# Colour tagging for the final 4K output. Frames reach FFmpeg as RGB-equivalent
# BGR, so the RGB -> YUV conversion must use the same matrix the file is tagged with.
_COLOR_TAG_ARGS = [
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-color_range", "tv",
]
_COLOR_CONVERT_FILTER = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"


@dataclass(frozen=True)
class EncoderSpec:
    """A resolved video encoder with its FFmpeg arguments."""

    codec: str
    args: tuple[str, ...]
    is_hevc: bool

    @property
    def container_tag_args(self) -> list[str]:
        """Extra tag so HEVC plays in QuickTime/Apple players."""
        return ["-tag:v", "hvc1"] if self.is_hevc else []


@functools.lru_cache(maxsize=None)
def encoder_works(codec: str, args: tuple[str, ...] = ()) -> bool:
    """Return True if FFmpeg can really encode one frame with ``codec`` and these ``args``.

    A listed encoder is not enough: e.g. ``hevc_nvenc`` is compiled in on many
    builds but fails at runtime without the NVIDIA encode library, and option or
    preset names differ between FFmpeg versions. Probing with the exact arguments
    we will use avoids a failure hours into a run.
    """
    codec_args = args if args else ("-c:v", codec)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=0.2",
        "-frames:v", "1", "-pix_fmt", "yuv420p",
        *codec_args, "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _preset_for(codec: str, preset: str) -> str:
    """Map the generic ``auto`` preset to a throughput-friendly per-codec default."""
    if preset != "auto":
        return preset
    return {"hevc_nvenc": "p5", "libx265": "veryfast", "libx264": "veryfast"}[codec]


def build_encoder_spec(codec: str, crf: int, preset: str = "auto") -> EncoderSpec:
    """Build FFmpeg encoder arguments for an explicit (non-auto) codec."""
    if codec not in CODEC_CHOICES or codec == "auto":
        raise ValueError(f"Unsupported video codec: {codec!r}")
    chosen = _preset_for(codec, preset)
    if codec == "hevc_nvenc":
        # NVENC CQ runs ~4 steps above an equivalent x264/x265 CRF.
        args = (
            "-c:v", "hevc_nvenc", "-preset", chosen, "-rc", "vbr",
            "-cq", str(min(51, crf + 4)), "-b:v", "0",
        )
        return EncoderSpec("hevc_nvenc", args, is_hevc=True)
    if codec == "libx265":
        args = ("-c:v", "libx265", "-preset", chosen, "-crf", str(crf), "-x265-params", "log-level=error")
        return EncoderSpec("libx265", args, is_hevc=True)
    args = ("-c:v", "libx264", "-preset", chosen, "-crf", str(crf))
    return EncoderSpec("libx264", args, is_hevc=False)


def resolve_encoder(preference: str, crf: int, preset: str = "auto") -> EncoderSpec:
    """Pick the best working encoder for ``preference`` (``auto`` tries GPU HEVC first)."""
    if preference not in CODEC_CHOICES:
        raise ValueError(f"Unsupported video codec: {preference!r}")
    candidates = ("hevc_nvenc", "libx265", "libx264") if preference == "auto" else (preference,)
    for codec in candidates:
        spec = build_encoder_spec(codec, crf, preset)
        if encoder_works(codec, spec.args):
            logger.info("Using video encoder %s", spec.codec)
            return spec
        logger.info("Video encoder %s is not usable here; trying next option.", codec)
    # Last resort keeps the pipeline moving even if probing itself failed.
    return build_encoder_spec("libx264", crf, preset)


def build_chunk_writer_cmd(
    output_file: Path,
    width: int,
    height: int,
    fps: float,
    encoder: EncoderSpec,
    scale_filter: str | None,
) -> list[str]:
    """FFmpeg command reading raw BGR frames on stdin and writing a final-quality chunk.

    ``-loglevel error -nostats`` keeps stderr nearly empty. Progress output
    otherwise fills the stderr pipe and blocks FFmpeg (and the whole worker).
    """
    filters = [f for f in (scale_filter, _COLOR_CONVERT_FILTER) if f]
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-vf", ",".join(filters),
        *encoder.args,
        "-pix_fmt", "yuv420p",
        *_COLOR_TAG_ARGS,
        *encoder.container_tag_args,
        str(output_file),
    ]


def build_legacy_chunk_writer_cmd(output_file: Path, width: int, height: int, fps: float) -> list[str]:
    """Original intermediate writer (libx264 CRF 16, no scaling) kept for legacy callers."""
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_file),
    ]
