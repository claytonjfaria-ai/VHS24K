"""VHS24K: VHS/DVD to 4K restoration and upscaling pipeline."""

from vhs24k.config import PipelineConfig
from vhs24k.postprocess import assemble_final, build_4k_scale_filter
from vhs24k.preprocess import (
    SourceGroup,
    VideoInfo,
    build_deinterlace_denoise_filter,
    discover_sources,
    prepare_chunks,
    probe_video,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "SourceGroup",
    "VideoInfo",
    "__version__",
    "assemble_final",
    "build_4k_scale_filter",
    "build_deinterlace_denoise_filter",
    "discover_sources",
    "prepare_chunks",
    "probe_video",
]
