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
from vhs24k.quality_review import (
    EnhancementRecipe,
    FrameQualityMetrics,
    analyze_frame,
    apply_overscan_crop,
    apply_post_upscale_4k_camera_polish,
    apply_pre_upscale_color_and_lighting,
    closed_loop_review_and_refine,
    compute_safe_overscan_crop,
    derive_enhancement_recipe,
    save_comparison_and_report,
)

__version__ = "0.1.0"

__all__ = [
    "EnhancementRecipe",
    "FrameQualityMetrics",
    "PipelineConfig",
    "SourceGroup",
    "VideoInfo",
    "__version__",
    "analyze_frame",
    "apply_overscan_crop",
    "apply_post_upscale_4k_camera_polish",
    "apply_pre_upscale_color_and_lighting",
    "assemble_final",
    "build_4k_scale_filter",
    "build_deinterlace_denoise_filter",
    "closed_loop_review_and_refine",
    "compute_safe_overscan_crop",
    "derive_enhancement_recipe",
    "discover_sources",
    "prepare_chunks",
    "probe_video",
    "save_comparison_and_report",
]
