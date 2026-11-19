"""Final 4K scaling, pillarboxing, audio muxing, and encoding."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from vhs24k.config import PipelineConfig
from vhs24k.encoding import EncoderSpec, build_encoder_spec, resolve_encoder

logger = logging.getLogger(__name__)

UHD_WIDTH = 3840
UHD_HEIGHT = 2160


def build_4k_scale_filter(
    target_mode: str,
    dar_width: int = 4,
    dar_height: int = 3,
) -> str:
    """Build the FFmpeg -vf filter string for 4K scaling and optional pillarboxing.

    - ``fullscreen_uhd``: scales to full-screen 3840x2160 with final 4K luma acutance and SAR=1.
    - ``pillarbox_uhd``: scales height to 2160 preserving DAR (e.g. 2880x2160 for 4:3)
      and pads to 3840x2160 centered with black bars, setting SAR=1.
    - ``native_4x``: scales to DAR-corrected 2160p (e.g. 2880x2160 for 4:3) with SAR=1.
    """
    if target_mode not in ("fullscreen_uhd", "pillarbox_uhd", "native_4x"):
        raise ValueError(f"Unsupported target_4k_mode: {target_mode!r}")
    if dar_width <= 0 or dar_height <= 0:
        raise ValueError(f"DAR dimensions must be positive, got {dar_width}:{dar_height}")

    if target_mode == "fullscreen_uhd":
        return f"scale={UHD_WIDTH}:{UHD_HEIGHT}:flags=lanczos,setsar=1"

    scaled_w = round(UHD_HEIGHT * dar_width / dar_height)
    if scaled_w % 2 != 0:
        scaled_w += 1

    if target_mode == "native_4x":
        return f"scale={scaled_w}:{UHD_HEIGHT}:flags=lanczos,setsar=1"

    if scaled_w < UHD_WIDTH:
        return (
            f"scale={scaled_w}:{UHD_HEIGHT}:flags=lanczos,"
            f"pad={UHD_WIDTH}:{UHD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )
    return f"scale={UHD_WIDTH}:{UHD_HEIGHT}:flags=lanczos,setsar=1"


def _parse_dar(dar_str: str | None) -> tuple[int, int]:
    """Parse a 'W:H' DAR string, defaulting to (4, 3)."""
    if not dar_str or ":" not in dar_str:
        return 4, 3
    parts = dar_str.split(":", 1)
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return 4, 3
    return (w, h) if w > 0 and h > 0 else (4, 3)


def _has_encoder(encoder_name: str) -> bool:
    """Check whether FFmpeg reports support for the given encoder."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return encoder_name in proc.stdout


_HEVC_CODECS = ("libx265", "hevc_nvenc")


def _uniform_final_encode(chunks: list[dict]) -> dict | None:
    """Return the shared ``final_encode`` tag if *every* chunk has the same one, else ``None``.

    Stream-copying needs identical codec, size and parameters across all chunks,
    so a single untagged or differently-tagged chunk forces the safe re-encode path.
    """
    tags = [c.get("final_encode") for c in chunks]
    if not tags or not all(isinstance(t, dict) and t.get("codec") for t in tags):
        return None
    first = tags[0]
    return first if all(t == first for t in tags) else None


def _build_stream_copy_command(
    concat_file: Path,
    audio_path: Path | None,
    output_path: Path,
    codec: str,
) -> list[str]:
    """Join already-final 4K chunks and mux audio without re-encoding the video."""
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
    cmd.extend(["-f", "concat", "-safe", "0", "-i", str(concat_file)])
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])
    cmd.extend(["-map", "0:v:0", "-c:v", "copy"])
    if codec in _HEVC_CODECS:
        cmd.extend(["-tag:v", "hvc1"])
    if audio_path is not None:
        cmd.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", str(output_path)])
    return cmd


def _build_assemble_command(
    concat_file: Path,
    audio_path: Path | None,
    vf_filter: str,
    encoder: EncoderSpec,
    output_path: Path,
) -> list[str]:
    """Construct the full re-encode assembly command (used when chunks are not already final)."""
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
    ]
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])

    cmd.extend(["-vf", vf_filter, *encoder.args, *encoder.container_tag_args])
    cmd.extend([
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
    ])

    if audio_path is not None:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    else:
        cmd.append("-an")

    cmd.extend(["-movflags", "+faststart", str(output_path)])
    return cmd


def assemble_final(manifest_path: Path, output_path: Path, cfg: PipelineConfig) -> Path:
    """Concatenate upscaled chunks, apply 4K scale/pad filter, mux audio, and encode."""
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    if not chunks and not cfg.dry_run:
        raise ValueError(f"Manifest has no chunks: {manifest_path}")

    incomplete = [c["index"] for c in chunks if c.get("status") != "done"]
    if incomplete and not cfg.dry_run:
        raise RuntimeError(
            f"Cannot assemble final video; chunks not done: {incomplete}"
        )

    upscaled_paths = [Path(c["output_path"]) for c in chunks]
    if not cfg.dry_run:
        missing = [str(p) for p in upscaled_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing upscaled chunk files: {missing}")

    concat_file = manifest_path.parent / "concat_upscaled.txt"
    concat_lines = [f"file '{p.resolve().as_posix()}'" for p in upscaled_paths]
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    raw_audio = manifest.get("audio_path")
    audio_path: Path | None = Path(raw_audio) if raw_audio else None
    if audio_path is not None and not cfg.dry_run and not audio_path.is_file():
        raise FileNotFoundError(f"Extracted audio track not found: {audio_path}")

    video_info = manifest.get("video_info") or {}
    dar_w, dar_h = _parse_dar(video_info.get("dar"))
    vf_filter = build_4k_scale_filter(cfg.target_4k_mode, dar_w, dar_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_encode = _uniform_final_encode(chunks)
    if final_encode is None and any(c.get("final_encode") for c in chunks):
        logger.warning(
            "Chunks were encoded differently (e.g. resumed from an older session); "
            "falling back to a full re-encode of the joined video."
        )
    if final_encode is not None:
        copy_cmd = _build_stream_copy_command(
            concat_file, audio_path, output_path, str(final_encode["codec"])
        )
        if cfg.dry_run:
            return output_path
        try:
            subprocess.run(copy_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FFmpeg stream-copy assembly failed: {exc.stderr.strip()}") from exc
        return output_path

    # Chunks are not final 4K: scale and encode the whole video here (slow, CPU-bound).
    if cfg.dry_run:
        preferred = cfg.video_codec if cfg.video_codec != "auto" else "libx265"
        primary = build_encoder_spec(preferred, cfg.crf, cfg.preset)
    else:
        primary = resolve_encoder(cfg.video_codec, cfg.crf, cfg.preset)
    cmd = _build_assemble_command(concat_file, audio_path, vf_filter, primary, output_path)

    if cfg.dry_run:
        return output_path

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        if primary.codec == "libx264":
            raise RuntimeError(f"FFmpeg assembly failed: {exc.stderr.strip()}") from exc
        fallback = build_encoder_spec("libx264", cfg.crf, cfg.preset)
        fallback_cmd = _build_assemble_command(concat_file, audio_path, vf_filter, fallback, output_path)
        try:
            subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as fallback_exc:
            raise RuntimeError(
                f"FFmpeg assembly failed with {primary.codec} and libx264: {fallback_exc.stderr.strip()}"
            ) from fallback_exc

    return output_path
