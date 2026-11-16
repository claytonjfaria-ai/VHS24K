"""Final 4K scaling, pillarboxing, audio muxing, and encoding."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vhs24k.config import PipelineConfig

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


def _build_assemble_command(
    concat_file: Path,
    audio_path: Path | None,
    vf_filter: str,
    codec: str,
    output_path: Path,
    cfg: PipelineConfig,
) -> list[str]:
    """Construct the final FFmpeg assembly command."""
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

    cmd.extend(["-vf", vf_filter, "-c:v", codec, "-crf", str(cfg.crf), "-preset", cfg.preset])
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
    primary_codec = "libx264"
    cmd = _build_assemble_command(concat_file, audio_path, vf_filter, primary_codec, output_path, cfg)

    if cfg.dry_run:
        return output_path

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        if primary_codec == "libx265":
            fallback_cmd = _build_assemble_command(
                concat_file, audio_path, vf_filter, "libx264", output_path, cfg
            )
            try:
                subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as fallback_exc:
                raise RuntimeError(
                    f"FFmpeg assembly failed with libx265 and libx264: {fallback_exc.stderr.strip()}"
                ) from fallback_exc
        else:
            raise RuntimeError(f"FFmpeg assembly failed: {exc.stderr.strip()}") from exc

    return output_path
