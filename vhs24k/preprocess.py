"""Source discovery, probing, deinterlacing, denoising, and chunking."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from vhs24k.config import PipelineConfig

MIN_SOURCE_BYTES = 1_000_000  # 1 MB threshold to skip menu/stub/corrupt files
_VOB_TITLE_RE = re.compile(r"^VTS_(\d+)_(\d+)\.vob$", re.IGNORECASE)
_INTERLACED_FIELD_ORDERS = frozenset({"tt", "bb", "tb", "bt"})


@dataclass
class VideoInfo:
    """Probed metadata for a video file."""

    path: Path
    width: int
    height: int
    fps: float
    duration: float
    is_interlaced: bool
    dar: str
    sar: str
    audio_codec: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-compatible dictionary."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class SourceGroup:
    """A logical video source consisting of one or more ordered files."""

    name: str
    files: list[Path]
    source_type: Literal["vob", "mp4"]


def _parse_fps(rate_str: str | None) -> float:
    """Parse an FFprobe fractional frame rate string (e.g. '30000/1001')."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        num, den = float(num_s), float(den_s)
        return num / den if den != 0 else 0.0
    return float(rate_str)


def _default_dar(width: int, height: int) -> str:
    """Compute a reduced aspect ratio string from pixel dimensions."""
    if width <= 0 or height <= 0:
        return "4:3"
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def probe_video(path: Path) -> VideoInfo:
    """Probe video stream metadata using ffprobe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc

    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams", [])
    fmt = payload.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    fps = _parse_fps(video_stream.get("avg_frame_rate")) or _parse_fps(
        video_stream.get("r_frame_rate")
    )
    raw_duration = video_stream.get("duration") or fmt.get("duration") or 0.0
    duration = float(raw_duration)

    field_order = str(video_stream.get("field_order", "")).lower()
    is_interlaced = field_order in _INTERLACED_FIELD_ORDERS

    dar = str(video_stream.get("display_aspect_ratio") or "")
    if not dar or dar == "0:1":
        dar = _default_dar(width, height)

    sar = str(video_stream.get("sample_aspect_ratio") or "")
    if not sar or sar == "0:1":
        sar = "1:1"

    audio_codec = audio_stream.get("codec_name") if audio_stream else None

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        is_interlaced=is_interlaced,
        dar=dar,
        sar=sar,
        audio_codec=audio_codec,
    )


def _discover_vob_groups(input_dir: Path) -> list[SourceGroup]:
    """Discover DVD VOB title groups, ignoring menus (*_0.VOB, VIDEO_TS.VOB) and stubs."""
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_file():
            continue
        match = _VOB_TITLE_RE.match(entry.name)
        if not match:
            continue
        title_num, part_num = match.group(1), int(match.group(2))
        if part_num == 0 or entry.stat().st_size < MIN_SOURCE_BYTES:
            continue
        title_key = f"VTS_{title_num}"
        grouped.setdefault(title_key, []).append((part_num, entry))

    groups: list[SourceGroup] = []
    for title_key in sorted(grouped):
        ordered_files = [path for _, path in sorted(grouped[title_key], key=lambda item: item[0])]
        groups.append(SourceGroup(name=title_key, files=ordered_files, source_type="vob"))
    return groups


def _discover_mp4_groups(input_dir: Path) -> list[SourceGroup]:
    """Discover MP4 sources, ignoring corrupt or stub files under 1 MB."""
    groups: list[SourceGroup] = []
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".mp4":
            continue
        if entry.stat().st_size < MIN_SOURCE_BYTES:
            continue
        groups.append(SourceGroup(name=entry.stem, files=[entry], source_type="mp4"))
    return groups


def discover_sources(input_dir: Path, mode: str = "auto") -> list[SourceGroup]:
    """Discover valid VOB title groups or MP4 files in input_dir."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if mode not in ("auto", "vob", "mp4"):
        raise ValueError(f"Unsupported discovery mode: {mode!r}")

    if mode == "vob":
        return _discover_vob_groups(input_dir)
    if mode == "mp4":
        return _discover_mp4_groups(input_dir)

    vob_groups = _discover_vob_groups(input_dir)
    if vob_groups:
        return vob_groups
    return _discover_mp4_groups(input_dir)


def build_deinterlace_denoise_filter(cfg: PipelineConfig) -> str:
    """Build the FFmpeg -vf filter string for deinterlacing and mild denoising."""
    filters: list[str] = []
    if cfg.deinterlace == "bwdif":
        filters.append("bwdif=mode=0:parity=auto:deint=all")
    elif cfg.deinterlace == "yadif":
        filters.append("yadif=mode=0:parity=auto:deint=all")
    elif cfg.deinterlace != "none":
        raise ValueError(f"Unsupported deinterlace mode: {cfg.deinterlace!r}")

    if cfg.denoise_strength > 0.0:
        s = cfg.denoise_strength
        luma_s = round(4.0 * s, 2)
        chroma_s = round(3.0 * s, 2)
        luma_t = round(6.0 * s, 2)
        chroma_t = round(4.5 * s, 2)
        filters.append(f"hqdn3d={luma_s}:{chroma_s}:{luma_t}:{chroma_t}")

    return ",".join(filters)


def _is_manifest_prepared(manifest_path: Path, cfg: PipelineConfig | None = None) -> bool:
    """Return True if an existing manifest has all prepared chunk files on disk."""
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if data.get("status") not in ("prepared", "done"):
        return False
    if cfg is not None:
        if data.get("test_clip_seconds") != cfg.test_clip_seconds:
            return False
        if cfg.test_clip_seconds is not None and data.get("test_start_seconds") != cfg.test_start_seconds:
            return False
    chunks = data.get("chunks")
    if not chunks:
        return False
    if not all(Path(c["input_path"]).is_file() for c in chunks):
        return False
    audio_path = data.get("audio_path")
    if audio_path and not Path(audio_path).is_file():
        return False
    return True


def _build_ffmpeg_input_args(source: SourceGroup, concat_file: Path) -> list[str]:
    """Build FFmpeg input arguments for single or multi-part sources."""
    if not source.files:
        raise ValueError(f"SourceGroup {source.name!r} has no files")
    if len(source.files) == 1:
        return ["-i", str(source.files[0])]
    if source.source_type == "vob":
        joined = "|".join(str(p) for p in source.files)
        return ["-i", f"concat:{joined}"]

    concat_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{p.resolve().as_posix()}'" for p in source.files]
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["-f", "concat", "-safe", "0", "-i", str(concat_file)]


def _run_ffmpeg(cmd: list[str]) -> None:
    """Execute an FFmpeg command and raise a clear error on failure."""
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg command failed ({' '.join(cmd)}): {exc.stderr.strip()}"
        ) from exc


def prepare_chunks(source: SourceGroup, work_dir: Path, cfg: PipelineConfig) -> Path:
    """Preprocess a SourceGroup into segmented video chunks and extracted audio.

    Writes and returns the path to ``manifest.json`` inside ``work_dir / source.name``.
    If already prepared and all artifacts exist, returns immediately.
    When ``cfg.dry_run`` is True, writes planned commands into ``manifest.json``
    without executing FFmpeg.
    """
    work_dir = Path(work_dir)
    source_dir = work_dir / source.name
    chunks_dir = source_dir / "chunks"
    upscaled_dir = source_dir / "upscaled"
    manifest_path = source_dir / "manifest.json"

    if not cfg.dry_run and _is_manifest_prepared(manifest_path, cfg):
        return manifest_path

    source_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    upscaled_dir.mkdir(parents=True, exist_ok=True)

    video_info: VideoInfo | None = None
    try:
        video_info = probe_video(source.files[0])
    except (FileNotFoundError, RuntimeError, ValueError):
        if not cfg.dry_run:
            raise

    input_args = _build_ffmpeg_input_args(source, source_dir / "concat_input.txt")
    vf_filter = build_deinterlace_denoise_filter(cfg)

    seek_args = ["-ss", str(cfg.test_start_seconds)] if (cfg.test_clip_seconds is not None and cfg.test_start_seconds > 0) else []
    video_cmd: list[str] = ["ffmpeg", "-y", *seek_args, *input_args]
    if cfg.test_clip_seconds is not None:
        video_cmd.extend(["-t", str(cfg.test_clip_seconds)])
    if vf_filter:
        video_cmd.extend(["-vf", vf_filter])
    video_cmd.extend([
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "10",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
    ])

    if cfg.test_clip_seconds is not None:
        video_cmd.append(str(chunks_dir / "chunk_0000.mp4"))
    else:
        video_cmd.extend([
            "-f",
            "segment",
            "-segment_time",
            str(cfg.chunk_seconds),
            "-reset_timestamps",
            "1",
            str(chunks_dir / "chunk_%04d.mp4"),
        ])

    has_audio = video_info is None or video_info.audio_codec is not None
    audio_path = source_dir / "audio.mka"
    audio_cmd: list[str] | None = None
    if has_audio:
        audio_cmd = ["ffmpeg", "-y", *seek_args, *input_args]
        if cfg.test_clip_seconds is not None:
            audio_cmd.extend(["-t", str(cfg.test_clip_seconds)])
        audio_cmd.extend(["-vn", "-c:a", "copy", str(audio_path)])

    manifest: dict[str, Any] = {
        "source_name": source.name,
        "source_type": source.source_type,
        "source_files": [str(p) for p in source.files],
        "video_info": video_info.to_dict() if video_info else None,
        "audio_path": str(audio_path) if audio_cmd else None,
        "test_clip_seconds": cfg.test_clip_seconds,
        "test_start_seconds": cfg.test_start_seconds if cfg.test_clip_seconds is not None else 0,
        "planned_commands": {
            "video": video_cmd,
            "audio": audio_cmd,
        },
        "chunks": [],
        "status": "dry_run" if cfg.dry_run else "pending",
    }

    if cfg.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    _run_ffmpeg(video_cmd)
    if audio_cmd is not None:
        _run_ffmpeg(audio_cmd)

    chunk_files = sorted(chunks_dir.glob("chunk_*.mp4"))
    if not chunk_files:
        raise RuntimeError(f"FFmpeg produced no chunks in {chunks_dir}")

    manifest["chunks"] = [
        {
            "index": idx,
            "input_path": str(chunk_path),
            "output_path": str(upscaled_dir / chunk_path.name),
            "status": "pending",
        }
        for idx, chunk_path in enumerate(chunk_files)
    ]
    manifest["status"] = "prepared"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
