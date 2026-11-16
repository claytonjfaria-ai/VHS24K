"""Configuration dataclass and validation for the VHS24K pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass
class PipelineConfig:
    """Configuration for VHS/DVD to 4K restoration and upscaling."""

    input_dir: Path
    work_dir: Path
    output_dir: Path
    source_mode: Literal["vob", "mp4", "auto"] = "auto"
    deinterlace: Literal["bwdif", "yadif", "none"] = "bwdif"
    denoise_strength: float = 0.35
    model_name: str = "realesr-general-x4v3"
    face_enhance: bool = True
    face_weight: float = 0.5
    tile_size: int = 512
    fp16: bool = True
    chunk_seconds: int = 60
    target_4k_mode: Literal["pillarbox_uhd", "native_4x"] = "pillarbox_uhd"
    crf: int = 16
    preset: str = "medium"
    num_gpus: int = 2
    test_clip_seconds: int | None = None
    realesrgan_dir: Path | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.work_dir = Path(self.work_dir)
        self.output_dir = Path(self.output_dir)
        if self.realesrgan_dir is not None:
            self.realesrgan_dir = Path(self.realesrgan_dir)

        if self.source_mode not in ("vob", "mp4", "auto"):
            raise ValueError(f"Unsupported source_mode: {self.source_mode!r}")
        if self.deinterlace not in ("bwdif", "yadif", "none"):
            raise ValueError(f"Unsupported deinterlace: {self.deinterlace!r}")
        if self.target_4k_mode not in ("pillarbox_uhd", "native_4x"):
            raise ValueError(f"Unsupported target_4k_mode: {self.target_4k_mode!r}")
        if not 0.0 <= self.denoise_strength <= 1.0:
            raise ValueError(f"denoise_strength must be in [0.0, 1.0], got {self.denoise_strength}")
        if not 0.0 <= self.face_weight <= 1.0:
            raise ValueError(f"face_weight must be in [0.0, 1.0], got {self.face_weight}")
        if not 0 <= self.crf <= 51:
            raise ValueError(f"crf must be in [0, 51], got {self.crf}")
        if self.chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds must be > 0, got {self.chunk_seconds}")
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {self.tile_size}")
        if self.num_gpus <= 0:
            raise ValueError(f"num_gpus must be >= 1, got {self.num_gpus}")
        if self.test_clip_seconds is not None and self.test_clip_seconds <= 0:
            raise ValueError(
                f"test_clip_seconds must be > 0 when set, got {self.test_clip_seconds}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        data = asdict(self)
        for key in ("input_dir", "work_dir", "output_dir", "realesrgan_dir"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data
