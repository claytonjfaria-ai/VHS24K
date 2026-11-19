"""Unit and integration tests for the VHS24K pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vhs24k import cli
from vhs24k.config import PipelineConfig
from vhs24k.postprocess import assemble_final, build_4k_scale_filter
from vhs24k.preprocess import (
    MIN_SOURCE_BYTES,
    SourceGroup,
    VideoInfo,
    _resolve_test_seek_and_input,
    build_deinterlace_denoise_filter,
    discover_sources,
    prepare_chunks,
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
    remove_vhs_white_line_dropouts,
    save_comparison_and_report,
    temporal_smooth_4k_frame,
)
from vhs24k.upscale import (
    build_enhancer_worker_cmd,
    build_realesrgan_cmd,
    upscale_manifest,
)


def _make_file(path: Path, size: int) -> Path:
    """Create a file with the specified byte size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        if size > 0:
            handle.seek(size - 1)
            handle.write(b"\0")
    return path


class TestPipelineConfig:
    def test_defaults_and_path_coercion(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(
            input_dir=str(tmp_path / "in"),  # type: ignore[arg-type]
            work_dir=str(tmp_path / "work"),  # type: ignore[arg-type]
            output_dir=str(tmp_path / "out"),  # type: ignore[arg-type]
            realesrgan_dir=str(tmp_path / "esrgan"),  # type: ignore[arg-type]
        )
        assert isinstance(cfg.input_dir, Path)
        assert isinstance(cfg.work_dir, Path)
        assert isinstance(cfg.output_dir, Path)
        assert isinstance(cfg.realesrgan_dir, Path)
        assert cfg.source_mode == "auto"
        assert cfg.deinterlace == "bwdif"
        assert cfg.denoise_strength == 0.65
        assert cfg.model_name == "realesr-general-x4v3"
        assert cfg.face_enhance is True
        assert cfg.face_weight == 0.85
        assert cfg.tile_size == 512
        assert cfg.fp16 is True
        assert cfg.chunk_seconds == 60
        assert cfg.target_4k_mode == "fullscreen_uhd"
        assert cfg.crf == 17
        assert cfg.preset == "medium"
        assert cfg.num_gpus == 2
        assert cfg.test_clip_seconds is None
        assert cfg.test_start_seconds == 900
        assert cfg.source_name is None
        assert cfg.dry_run is False
        assert cfg.quality_review is True
        assert cfg.target_quality_score == 82.0

        serialized = cfg.to_dict()
        assert serialized["input_dir"] == str(tmp_path / "in")
        assert json.dumps(serialized)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"source_mode": "avi"},
            {"deinterlace": "bob"},
            {"target_4k_mode": "stretch"},
            {"denoise_strength": -0.1},
            {"denoise_strength": 1.1},
            {"face_weight": -0.5},
            {"face_weight": 1.5},
            {"crf": -1},
            {"crf": 52},
            {"chunk_seconds": 0},
            {"tile_size": 0},
            {"num_gpus": 0},
            {"test_clip_seconds": 0},
            {"target_quality_score": -0.1},
            {"target_quality_score": 100.1},
        ],
    )
    def test_validation_errors(self, tmp_path: Path, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            PipelineConfig(
                input_dir=tmp_path / "in",
                work_dir=tmp_path / "work",
                output_dir=tmp_path / "out",
                **kwargs,
            )


class TestDiscoverSources:
    def test_vob_title_grouping_skips_menus_and_stubs(self, tmp_path: Path) -> None:
        valid_size = MIN_SOURCE_BYTES + 1024
        stub_size = MIN_SOURCE_BYTES - 1

        _make_file(tmp_path / "VIDEO_TS.VOB", valid_size)
        _make_file(tmp_path / "VTS_01_0.VOB", valid_size)
        vts_01_1 = _make_file(tmp_path / "VTS_01_1.VOB", valid_size)
        _make_file(tmp_path / "VTS_02_0.VOB", valid_size)
        vts_02_1 = _make_file(tmp_path / "VTS_02_1.VOB", valid_size)
        vts_02_2 = _make_file(tmp_path / "VTS_02_2.VOB", valid_size)
        vts_02_3 = _make_file(tmp_path / "VTS_02_3.VOB", valid_size)
        _make_file(tmp_path / "VTS_03_1.VOB", stub_size)

        groups = discover_sources(tmp_path, mode="vob")
        assert [g.name for g in groups] == ["VTS_01", "VTS_02"]
        assert groups[0].source_type == "vob"
        assert groups[0].files == [vts_01_1]
        assert groups[1].source_type == "vob"
        assert groups[1].files == [vts_02_1, vts_02_2, vts_02_3]

        # auto mode prefers VOB when present
        auto_groups = discover_sources(tmp_path, mode="auto")
        assert auto_groups == groups

    def test_mp4_discovery_and_auto_fallback(self, tmp_path: Path) -> None:
        valid_size = MIN_SOURCE_BYTES + 1024
        stub_size = MIN_SOURCE_BYTES - 1

        w1 = _make_file(tmp_path / "Wedding1.mp4", valid_size)
        w2 = _make_file(tmp_path / "Wedding2.mp4", valid_size)
        _make_file(tmp_path / "corrupt_stub.mp4", stub_size)

        mp4_groups = discover_sources(tmp_path, mode="mp4")
        assert [g.name for g in mp4_groups] == ["Wedding1", "Wedding2"]
        assert all(g.source_type == "mp4" for g in mp4_groups)
        assert mp4_groups[0].files == [w1]
        assert mp4_groups[1].files == [w2]

        auto_groups = discover_sources(tmp_path, mode="auto")
        assert auto_groups == mp4_groups

    def test_discover_invalid_dir_or_mode(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            discover_sources(tmp_path / "missing")
        with pytest.raises(ValueError):
            discover_sources(tmp_path, mode="mkv")


class TestFilters:
    def test_deinterlace_denoise_filter(self, tmp_path: Path) -> None:
        cfg_bwdif = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
            deinterlace="bwdif",
            denoise_strength=0.65,
        )
        assert (
            build_deinterlace_denoise_filter(cfg_bwdif)
            == "bwdif=mode=0:parity=auto:deint=all,"
            "atadenoise=0a=0.06:0b=0.12:1a=0.06:1b=0.12:2a=0.06:2b=0.12,"
            "deflicker=size=5:mode=am,"
            "hqdn3d=2.27:2.93:3.9:3.9"
        )

        cfg_yadif = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
            deinterlace="yadif",
            denoise_strength=0.0,
        )
        assert build_deinterlace_denoise_filter(cfg_yadif) == "yadif=mode=0:parity=auto:deint=all"

        cfg_none = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
            deinterlace="none",
            denoise_strength=0.0,
        )
        assert build_deinterlace_denoise_filter(cfg_none) == ""

    def test_4k_scale_filter_modes(self) -> None:
        assert (
            build_4k_scale_filter("fullscreen_uhd", 4, 3)
            == "scale=3840:2160:flags=lanczos,setsar=1"
        )
        assert build_4k_scale_filter("pillarbox_uhd", 4, 3) == (
            "scale=2880:2160:flags=lanczos,"
            "pad=3840:2160:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1"
        )
        assert (
            build_4k_scale_filter("native_4x", 4, 3)
            == "scale=2880:2160:flags=lanczos,setsar=1"
        )
        assert (
            build_4k_scale_filter("pillarbox_uhd", 16, 9)
            == "scale=3840:2160:flags=lanczos,setsar=1"
        )

        with pytest.raises(ValueError):
            build_4k_scale_filter("unknown", 4, 3)
        with pytest.raises(ValueError):
            build_4k_scale_filter("pillarbox_uhd", 0, 3)


class TestRealESRGANCommand:
    def test_realesrgan_cmd_flags(self, tmp_path: Path) -> None:
        esrgan_dir = tmp_path / "Real-ESRGAN"
        chunk_in = tmp_path / "chunks" / "chunk_0000.mp4"
        chunk_out = tmp_path / "upscaled"

        cfg_default = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
            model_name="realesr-general-x4v3",
            denoise_strength=0.35,
            face_enhance=True,
            fp16=True,
            tile_size=512,
        )
        cmd_default = build_realesrgan_cmd(chunk_in, chunk_out, cfg_default, esrgan_dir)
        assert "-dn" in cmd_default
        assert cmd_default[cmd_default.index("-dn") + 1] == "0.35"
        assert "--face_enhance" in cmd_default
        assert "--fp32" not in cmd_default
        assert "--tile" in cmd_default
        assert cmd_default[cmd_default.index("--tile") + 1] == "512"

        cfg_alt = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
            model_name="RealESRGAN_x4plus",
            face_enhance=False,
            fp16=False,
            tile_size=256,
        )
        cmd_alt = build_realesrgan_cmd(chunk_in, chunk_out, cfg_alt, esrgan_dir)
        assert "-dn" not in cmd_alt
        assert "--face_enhance" not in cmd_alt
        assert "--fp32" in cmd_alt
        assert "--tile" in cmd_alt
        assert cmd_alt[cmd_alt.index("--tile") + 1] == "256"

    def test_enhancer_worker_cmd_flags(self, tmp_path: Path) -> None:
        esrgan_dir = tmp_path / "Real-ESRGAN"
        chunk_in = tmp_path / "chunks" / "chunk_0000.mp4"
        chunk_out_file = tmp_path / "upscaled" / "chunk_0000.mp4"
        report_dir = tmp_path / "reports"

        cfg_default = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
        )
        cmd_default = build_enhancer_worker_cmd(
            chunk_in, chunk_out_file, cfg_default, esrgan_dir, report_dir=report_dir
        )
        assert "vhs24k.enhancer_worker" in cmd_default
        assert cmd_default[cmd_default.index("-n") + 1] == "realesr-general-x4v3"
        assert cmd_default[cmd_default.index("--dn") + 1] == "0.65"
        assert cmd_default[cmd_default.index("--face-weight") + 1] == "0.85"
        assert cmd_default[cmd_default.index("--target-quality-score") + 1] == "82.0"
        assert "--face-enhance" in cmd_default
        assert "--fp32" not in cmd_default
        assert "--no-quality-review" not in cmd_default
        assert cmd_default[cmd_default.index("--report-dir") + 1] == str(report_dir)

        cfg_disabled = PipelineConfig(
            input_dir=tmp_path,
            work_dir=tmp_path,
            output_dir=tmp_path,
            face_enhance=False,
            fp16=False,
            quality_review=False,
            face_weight=0.4,
            target_quality_score=75.0,
        )
        cmd_disabled = build_enhancer_worker_cmd(
            chunk_in, chunk_out_file, cfg_disabled, esrgan_dir, report_dir=None
        )
        assert "--face-enhance" not in cmd_disabled
        assert "--fp32" in cmd_disabled
        assert "--no-quality-review" in cmd_disabled
        assert "--report-dir" not in cmd_disabled
        assert cmd_disabled[cmd_disabled.index("--face-weight") + 1] == "0.4"
        assert cmd_disabled[cmd_disabled.index("--target-quality-score") + 1] == "75.0"


class TestManifestCompatibilityAndDryRun:
    def test_prepare_chunks_dry_run(self, tmp_path: Path) -> None:
        vob1 = _make_file(tmp_path / "in" / "VTS_02_1.VOB", MIN_SOURCE_BYTES + 10)
        vob2 = _make_file(tmp_path / "in" / "VTS_02_2.VOB", MIN_SOURCE_BYTES + 10)
        source = SourceGroup(name="VTS_02", files=[vob1, vob2], source_type="vob")

        cfg = PipelineConfig(
            input_dir=tmp_path / "in",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "out",
            test_clip_seconds=15,
            dry_run=True,
        )
        manifest_path = prepare_chunks(source, cfg.work_dir, cfg)
        assert manifest_path == cfg.work_dir / "VTS_02" / "manifest.json"
        assert manifest_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_name"] == "VTS_02"
        assert manifest["source_type"] == "vob"
        assert manifest["status"] == "dry_run"
        assert manifest["chunks"] == []
        assert "-t" in manifest["planned_commands"]["video"]
        assert "15" in manifest["planned_commands"]["video"]
        assert f"concat:{vob1}|{vob2}" in manifest["planned_commands"]["video"]

    def test_resolve_test_seek_and_input_multifile_vob(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vob1 = _make_file(tmp_path / "in" / "VTS_02_1.VOB", MIN_SOURCE_BYTES + 10)
        vob2 = _make_file(tmp_path / "in" / "VTS_02_2.VOB", MIN_SOURCE_BYTES + 10)
        vob3 = _make_file(tmp_path / "in" / "VTS_02_3.VOB", MIN_SOURCE_BYTES + 10)
        source = SourceGroup(name="VTS_02", files=[vob1, vob2, vob3], source_type="vob")
        concat_file = tmp_path / "work" / "VTS_02" / "concat_input.txt"

        cfg = PipelineConfig(
            input_dir=tmp_path / "in",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "out",
            test_clip_seconds=15,
            test_start_seconds=1200,
            dry_run=False,
        )

        # Case 1: Duration walk across VOBs (600s + 700s + 500s -> target 1200s lands at +600s in vob2)
        durations = {vob1: 600.0, vob2: 700.0, vob3: 500.0}

        def fake_probe(path: Path) -> VideoInfo:
            return VideoInfo(
                path=path,
                width=720,
                height=480,
                fps=29.97,
                duration=durations[path],
                is_interlaced=True,
                dar="4:3",
                sar="8:9",
                audio_codec="ac3",
            )

        monkeypatch.setattr("vhs24k.preprocess.probe_video", fake_probe)
        seek_args, input_args = _resolve_test_seek_and_input(source, cfg, concat_file)
        assert seek_args == ["-ss", "600"]
        assert input_args == ["-i", str(vob2)]

        # Case 2: Probe failure falls back to middle file with -ss 180
        def failing_probe(path: Path) -> VideoInfo:
            raise RuntimeError("ffprobe unavailable")

        monkeypatch.setattr("vhs24k.preprocess.probe_video", failing_probe)
        fb_seek, fb_input = _resolve_test_seek_and_input(source, cfg, concat_file)
        assert fb_seek == ["-ss", "180"]
        assert fb_input == ["-i", str(vob2)]

    def test_upscale_and_postprocess_manifest_schema(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "work" / "VTS_01"
        chunks_dir = source_dir / "chunks"
        upscaled_dir = source_dir / "upscaled"
        chunks_dir.mkdir(parents=True)
        upscaled_dir.mkdir(parents=True)

        chunk_in = _make_file(chunks_dir / "chunk_0000.mp4", 1024)
        chunk_out = upscaled_dir / "chunk_0000.mp4"
        audio_path = _make_file(source_dir / "audio.mka", 512)

        manifest_path = source_dir / "manifest.json"
        manifest = {
            "source_name": "VTS_01",
            "source_type": "vob",
            "source_files": [str(tmp_path / "in" / "VTS_01_1.VOB")],
            "video_info": {"dar": "4:3"},
            "audio_path": str(audio_path),
            "planned_commands": {"video": [], "audio": []},
            "chunks": [
                {
                    "index": 0,
                    "input_path": str(chunk_in),
                    "output_path": str(chunk_out),
                    "status": "pending",
                }
            ],
            "status": "prepared",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        cfg = PipelineConfig(
            input_dir=tmp_path / "in",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "out",
            dry_run=True,
        )

        # upscale_manifest accepts preprocess.py manifest schema in dry_run mode
        returned_manifest = upscale_manifest(manifest_path, cfg)
        assert returned_manifest == manifest_path

        # Simulate completed upscale and verify postprocess.assemble_final compatibility
        _make_file(chunk_out, 2048)
        manifest["chunks"][0]["status"] = "done"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        out_mp4 = tmp_path / "out" / "VTS_01_4k.mp4"
        result = assemble_final(manifest_path, out_mp4, cfg)
        assert result == out_mp4
        concat_file = source_dir / "concat_upscaled.txt"
        assert concat_file.is_file()
        assert chunk_out.resolve().as_posix() in concat_file.read_text(encoding="utf-8")


class TestCLI:
    def test_argument_parsing(self, tmp_path: Path) -> None:
        parser = cli.build_parser()
        default_args = parser.parse_args(["run"])
        default_cfg = cli._config_from_args(default_args)
        assert default_cfg.model_name == "realesr-general-x4v3"
        assert default_cfg.denoise_strength == 0.65
        assert default_cfg.face_weight == 0.85
        assert default_cfg.target_4k_mode == "fullscreen_uhd"
        assert default_cfg.test_start_seconds == 900
        assert default_cfg.source_name is None

        args = parser.parse_args(
            [
                "run",
                "--input-dir",
                str(tmp_path / "in"),
                "--work-dir",
                str(tmp_path / "work"),
                "--output-dir",
                str(tmp_path / "out"),
                "--source-mode",
                "vob",
                "--source-name",
                "VTS_02",
                "--model",
                "RealESRGAN_x4plus",
                "--denoise-strength",
                "0.45",
                "--no-face-enhance",
                "--face-weight",
                "0.65",
                "--no-quality-review",
                "--target-quality-score",
                "88.5",
                "--chunk-seconds",
                "30",
                "--test-clip",
                "15",
                "--test-start",
                "1200",
                "--num-gpus",
                "1",
                "--4k-mode",
                "native_4x",
                "--dry-run",
            ]
        )
        cfg = cli._config_from_args(args)
        assert args.command == "run"
        assert cfg.source_mode == "vob"
        assert cfg.source_name == "VTS_02"
        assert cfg.model_name == "RealESRGAN_x4plus"
        assert cfg.denoise_strength == 0.45
        assert cfg.face_enhance is False
        assert cfg.face_weight == 0.65
        assert cfg.quality_review is False
        assert cfg.target_quality_score == 88.5
        assert cfg.chunk_seconds == 30
        assert cfg.test_clip_seconds == 15
        assert cfg.test_start_seconds == 1200
        assert cfg.num_gpus == 1
        assert cfg.target_4k_mode == "native_4x"
        assert cfg.dry_run is True

    def test_cli_run_dry_run_smoke(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "in"
        work_dir = tmp_path / "work"
        out_dir = tmp_path / "out"
        _make_file(in_dir / "VTS_01_1.VOB", MIN_SOURCE_BYTES + 1024)

        exit_code = cli.main(
            [
                "run",
                "--input-dir",
                str(in_dir),
                "--work-dir",
                str(work_dir),
                "--output-dir",
                str(out_dir),
                "--test-clip",
                "15",
                "--dry-run",
            ]
        )
        assert exit_code == 0
        manifest_path = work_dir / "VTS_01" / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "dry_run"


def _make_synthetic_vhs_frame(height: int = 120, width: int = 160) -> np.ndarray:
    """Create a synthetic BGR frame with warm VHS color cast and compressed dynamic range."""
    rng = np.random.default_rng(42)
    yy = np.linspace(60, 140, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 40, width, dtype=np.float32)[None, :]
    base = yy + xx
    # Strong warm/yellow cast (low blue, high red)
    b = np.clip(base * 0.55 + rng.normal(0, 2.0, (height, width)), 0, 255)
    g = np.clip(base * 0.90 + rng.normal(0, 2.0, (height, width)), 0, 255)
    r = np.clip(base * 1.15 + rng.normal(0, 2.0, (height, width)), 0, 255)
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def _make_textured_frame(height: int = 120, width: int = 160) -> np.ndarray:
    """Create a high-contrast, sharp, full-dynamic-range BGR frame with no deficiencies."""
    yy, xx = np.mgrid[0:height, 0:width]
    ramp = (yy / max(height - 1, 1)) * 120.0 + 40.0
    fine = (((yy // 2) + (xx // 2)) % 2) * 75.0 - 37.5
    base = np.clip(ramp + fine, 20, 235)
    # Smooth, low-frequency color variation (zero chroma noise, colorfulness > 25, zero cast)
    color_shift_x = np.sin(2.0 * np.pi * xx / max(width, 1)) * 28.0
    color_shift_y = np.sin(2.0 * np.pi * yy / max(height, 1)) * 28.0
    b = np.clip(base + color_shift_x, 0, 255).astype(np.uint8)
    g = np.clip(base, 0, 255).astype(np.uint8)
    r = np.clip(base + color_shift_y, 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


class TestQualityReview:
    def test_analyze_frame_metrics_and_deficiencies(self) -> None:
        vhs_frame = _make_synthetic_vhs_frame()
        metrics = analyze_frame(vhs_frame)
        assert isinstance(metrics, FrameQualityMetrics)
        assert 0.0 <= metrics.overall_score <= 100.0
        assert 0.0 <= metrics.sharpness <= 100.0
        assert 0.0 <= metrics.face_clarity <= 100.0
        assert 0.0 <= metrics.dynamic_range <= 100.0
        assert 0.0 <= metrics.color_naturalness <= 100.0
        assert 0.0 <= metrics.vhs_artifact_free <= 100.0
        assert metrics.grade
        assert "vhs_color_cast" in metrics.deficiencies

        with pytest.raises(TypeError):
            analyze_frame([[1, 2, 3]])  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            analyze_frame(np.zeros((10, 10), dtype=np.uint8))

    def test_derive_enhancement_recipe_adapts_to_deficiencies(self) -> None:
        weak_metrics = FrameQualityMetrics(
            sharpness=50.0,
            face_clarity=55.0,
            dynamic_range=50.0,
            color_naturalness=50.0,
            vhs_artifact_free=60.0,
            overall_score=53.0,
            grade="C (Needs Re-Enhancement)",
            deficiencies=[
                "soft_scene_textures",
                "soft_faces",
                "compressed_dynamic_range",
                "vhs_color_cast",
                "muted_saturation",
                "chroma_noise",
            ],
        )
        recipe = derive_enhancement_recipe(weak_metrics, base_face_weight=0.75)
        assert recipe.wb_strength == 0.45
        assert recipe.clahe_clip_limit == 1.4
        assert recipe.vibrance_boost == 0.24
        assert recipe.chroma_denoise == 0.75
        assert recipe.micro_contrast == 0.65
        assert recipe.pre_sharpen == 0.0
        assert recipe.face_weight == 0.88

        clean_metrics = FrameQualityMetrics(
            sharpness=90.0,
            face_clarity=88.0,
            dynamic_range=88.0,
            color_naturalness=90.0,
            vhs_artifact_free=92.0,
            overall_score=89.6,
            grade="A+ (Modern 4K Camera)",
            deficiencies=[],
        )
        clean_recipe = derive_enhancement_recipe(clean_metrics, base_face_weight=0.75)
        assert clean_recipe.wb_strength == 0.30
        assert clean_recipe.clahe_clip_limit == 1.15
        assert clean_recipe.face_weight == 0.75

    def test_pre_upscale_color_and_lighting_and_post_polish(self) -> None:
        vhs_frame = _make_synthetic_vhs_frame()
        metrics = analyze_frame(vhs_frame)
        recipe = derive_enhancement_recipe(metrics)

        pre_fixed = apply_pre_upscale_color_and_lighting(vhs_frame, recipe)
        assert pre_fixed.shape == vhs_frame.shape
        assert pre_fixed.dtype == np.uint8

        pre_metrics = analyze_frame(pre_fixed)
        assert pre_metrics.dynamic_range >= metrics.dynamic_range
        # White balance reduces the red-blue VHS color cast spread
        orig_cast = abs(float(np.mean(vhs_frame[:, :, 2])) - float(np.mean(vhs_frame[:, :, 0])))
        fixed_cast = abs(float(np.mean(pre_fixed[:, :, 2])) - float(np.mean(pre_fixed[:, :, 0])))
        assert fixed_cast < orig_cast

        polished = apply_post_upscale_4k_camera_polish(pre_fixed, recipe)
        assert polished.shape == pre_fixed.shape
        assert polished.dtype == np.uint8

    def test_post_upscale_polish_edge_gate_preserves_flat_surfaces(self) -> None:
        # Left half: flat sky/water with tiny ±2 micro-noise (grad < 3.5 -> edge_gate == 0)
        # Right half: sharp structural step edge (grad > 15.5 -> edge_gate == 1)
        h, w = 80, 160
        rng = np.random.default_rng(7)
        frame = np.full((h, w, 3), 120, dtype=np.uint8)
        flat_noise = rng.integers(-2, 3, size=(h, 70, 1))
        frame[:, :70, :] = np.clip(120 + flat_noise, 0, 255).astype(np.uint8)
        frame[:, 110:, :] = 200  # strong vertical structural edge at x=110

        recipe = EnhancementRecipe(micro_contrast=0.65, second_pass_sharpen=0.0)
        polished = apply_post_upscale_4k_camera_polish(frame, recipe)

        # Flat region should be smoothed (std <= original std), not noise-amplified
        flat_before_std = float(np.std(frame[:, 10:60, 0].astype(np.float32)))
        flat_after_std = float(np.std(polished[:, 10:60, 0].astype(np.float32)))
        assert flat_after_std <= flat_before_std + 0.25

        # Structural edge at x=110 should gain local contrast (overshoot/undershoot acutance)
        edge_before_span = int(frame[40, 112, 0]) - int(frame[40, 108, 0])
        edge_after_span = int(polished[40, 112, 0]) - int(polished[40, 108, 0])
        assert edge_after_span >= edge_before_span

    def test_closed_loop_review_and_refine(self) -> None:
        before = _make_synthetic_vhs_frame()
        recipe = EnhancementRecipe()

        # Soft after-frame below target_score=95 triggers second-pass refinement
        out_frame, before_m, after_m, updated_recipe, refined = closed_loop_review_and_refine(
            before, before, recipe, target_score=95.0
        )
        assert refined is True
        assert out_frame.shape == before.shape
        assert isinstance(before_m, FrameQualityMetrics)
        assert isinstance(after_m, FrameQualityMetrics)
        # The returned recipe is never one that scored worse than the original
        assert after_m.overall_score >= analyze_frame(before).overall_score

        # A candidate that scores better is accepted and its recipe is returned
        better = _make_textured_frame(height=before.shape[0], width=before.shape[1])
        _, _, better_m, accepted_recipe, refined_better = closed_loop_review_and_refine(
            before,
            before,
            recipe,
            target_score=99.0,
            polish_fn=lambda _frame, _recipe: better,
        )
        assert refined_better is True
        assert accepted_recipe.second_pass_sharpen > 0.0
        assert better_m.overall_score >= analyze_frame(before).overall_score

        # A candidate that scores worse is rejected and the original recipe is kept
        worse = np.full_like(before, 128)
        kept_frame, _, kept_m, kept_recipe, _ = closed_loop_review_and_refine(
            before,
            before,
            recipe,
            target_score=99.0,
            polish_fn=lambda _frame, _recipe: worse,
        )
        assert kept_recipe == recipe
        assert kept_recipe.second_pass_sharpen == 0.0
        assert kept_frame is before
        assert kept_m.overall_score == analyze_frame(before).overall_score

        # Sharp after-frame above low target_score skips second-pass refinement
        sharp = _make_textured_frame()
        out_sharp, _, sharp_m, unchanged_recipe, refined_sharp = closed_loop_review_and_refine(
            before, sharp, recipe, target_score=50.0
        )
        assert sharp_m.sharpness >= 75.0
        assert refined_sharp is False
        assert unchanged_recipe.second_pass_sharpen == 0.0
        assert out_sharp is sharp

    def test_save_comparison_and_report(self, tmp_path: Path) -> None:
        before = _make_synthetic_vhs_frame()
        after = _make_textured_frame(height=240, width=320)
        before_m = analyze_frame(before)
        after_m = analyze_frame(after)
        recipe = EnhancementRecipe()

        saved = save_comparison_and_report(
            before_bgr=before,
            after_bgr=after,
            before_metrics=before_m,
            after_metrics=after_m,
            recipe=recipe,
            output_dir=tmp_path / "reports",
            prefix="chunk_0000",
        )
        report_path = Path(saved["report_json"])
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["before"]["overall_score"] == before_m.overall_score
        assert payload["after"]["overall_score"] == after_m.overall_score
        assert payload["grade"] == after_m.grade
        assert "recipe" in payload
        if "comparison_jpg" in saved:
            assert Path(saved["comparison_jpg"]).is_file()

    def test_overscan_crop_trims_black_borders_conservatively(self) -> None:
        h, w = 480, 720
        frame = np.full((h, w, 3), 140, dtype=np.uint8)
        # Simulate black VHS overscan edges & bottom head-switching noise bar
        frame[:, :20, :] = 0
        frame[:, -22:, :] = 0
        frame[:12, :, :] = 0
        frame[-20:, :, :] = 0

        y0, y1, x0, x1 = compute_safe_overscan_crop([frame])
        assert x0 > 0 and x0 <= int(w * 0.045) + 1
        assert (w - x1) > 0 and (w - x1) <= int(w * 0.045) + 1
        assert y0 > 0 and y0 <= int(h * 0.035) + 1
        assert (h - y1) > 0 and (h - y1) <= int(h * 0.055) + 1

        cropped = apply_overscan_crop(frame, (y0, y1, x0, x1))
        ch, cw = cropped.shape[:2]
        assert ch % 2 == 0 and cw % 2 == 0
        assert ch < h and cw < w
        assert apply_overscan_crop(frame, None).shape == frame.shape

    def test_remove_vhs_white_line_dropouts(self) -> None:
        h, w = 80, 160
        prev_bgr = np.full((h, w, 3), 90, dtype=np.uint8)
        next_bgr = np.full((h, w, 3), 90, dtype=np.uint8)
        curr_bgr = np.full((h, w, 3), 90, dtype=np.uint8)

        # Clean frame should report zero dropouts
        clean_out, n_clean = remove_vhs_white_line_dropouts(curr_bgr, prev_bgr, next_bgr)
        assert n_clean == 0
        assert np.array_equal(clean_out, curr_bgr)

        # Inject a bright horizontal white VHS tape dropout line
        curr_bgr[30, 20:140, :] = 255
        fixed_bgr, n_dropouts = remove_vhs_white_line_dropouts(curr_bgr, prev_bgr, next_bgr)
        assert n_dropouts > 0
        assert fixed_bgr.shape == curr_bgr.shape
        assert fixed_bgr.dtype == np.uint8
        assert float(np.mean(fixed_bgr[30, 20:140, :])) < 110.0

    def test_temporal_smooth_4k_frame(self) -> None:
        h, w = 64, 64
        prev_4k = np.full((h, w, 3), 100, dtype=np.uint8)
        curr_4k = np.full((h, w, 3), 104, dtype=np.uint8)  # small stationary micro-shimmer (diff=4)
        curr_4k[16:32, 16:32, :] = 220  # fast-moving bright edge (diff=120)

        # First frame (prev=None) passes through unchanged
        assert np.array_equal(temporal_smooth_4k_frame(curr_4k, None, strength=0.30), curr_4k)

        smoothed = temporal_smooth_4k_frame(curr_4k, prev_4k, strength=0.30)
        assert smoothed.shape == curr_4k.shape
        assert smoothed.dtype == np.uint8
        # Stationary shimmer is pulled toward prev_4k (100 < value < 104)
        assert int(smoothed[0, 0, 0]) < 104
        # Fast-moving edge is preserved without ghosting
        assert int(smoothed[24, 24, 0]) == 220
