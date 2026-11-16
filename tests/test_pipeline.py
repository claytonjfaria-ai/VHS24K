"""Unit and integration tests for the VHS24K pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vhs24k import cli
from vhs24k.config import PipelineConfig
from vhs24k.postprocess import assemble_final, build_4k_scale_filter
from vhs24k.preprocess import (
    MIN_SOURCE_BYTES,
    SourceGroup,
    build_deinterlace_denoise_filter,
    discover_sources,
    prepare_chunks,
)
from vhs24k.upscale import build_realesrgan_cmd, upscale_manifest


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
        assert cfg.denoise_strength == 0.35
        assert cfg.model_name == "realesr-general-x4v3"
        assert cfg.face_enhance is True
        assert cfg.face_weight == 0.5
        assert cfg.tile_size == 512
        assert cfg.fp16 is True
        assert cfg.chunk_seconds == 60
        assert cfg.target_4k_mode == "pillarbox_uhd"
        assert cfg.crf == 16
        assert cfg.preset == "medium"
        assert cfg.num_gpus == 2
        assert cfg.test_clip_seconds is None
        assert cfg.dry_run is False

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
            denoise_strength=0.35,
        )
        assert (
            build_deinterlace_denoise_filter(cfg_bwdif)
            == "bwdif=mode=0:parity=auto:deint=all,hqdn3d=1.4:1.05:2.1:1.57"
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
                "--model",
                "RealESRGAN_x4plus",
                "--no-face-enhance",
                "--chunk-seconds",
                "30",
                "--test-clip",
                "15",
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
        assert cfg.model_name == "RealESRGAN_x4plus"
        assert cfg.face_enhance is False
        assert cfg.chunk_seconds == 30
        assert cfg.test_clip_seconds == 15
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
