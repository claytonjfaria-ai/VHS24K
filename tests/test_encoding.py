"""Encoder selection and FFmpeg command construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from vhs24k import encoding
from vhs24k.encoding import (
    build_chunk_writer_cmd,
    build_encoder_spec,
    build_legacy_chunk_writer_cmd,
    resolve_encoder,
)


def _only_working(monkeypatch: pytest.MonkeyPatch, working: set[str]) -> None:
    monkeypatch.setattr(encoding, "encoder_works", lambda codec, args=(): codec in working)


def test_auto_prefers_gpu_hevc_when_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_working(monkeypatch, {"hevc_nvenc", "libx265", "libx264"})
    spec = resolve_encoder("auto", crf=17)
    assert spec.codec == "hevc_nvenc"
    assert spec.is_hevc
    assert spec.args[spec.args.index("-cq") + 1] == "21"


def test_auto_falls_back_to_x265_then_x264(monkeypatch: pytest.MonkeyPatch) -> None:
    _only_working(monkeypatch, {"libx265", "libx264"})
    assert resolve_encoder("auto", crf=17).codec == "libx265"
    _only_working(monkeypatch, {"libx264"})
    assert resolve_encoder("auto", crf=17).codec == "libx264"


def test_explicit_codec_that_does_not_work_still_returns_a_usable_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _only_working(monkeypatch, set())
    assert resolve_encoder("hevc_nvenc", crf=17).codec == "libx264"


def test_auto_preset_is_fast_per_codec() -> None:
    assert build_encoder_spec("libx265", 17).args[3] == "veryfast"
    assert build_encoder_spec("hevc_nvenc", 17).args[3] == "p5"
    assert build_encoder_spec("libx264", 17, "slow").args[3] == "slow"


def test_unknown_codec_rejected() -> None:
    with pytest.raises(ValueError):
        build_encoder_spec("prores", 17)
    with pytest.raises(ValueError):
        resolve_encoder("av1", 17)


def test_chunk_writer_cmd_is_final_quality_and_never_blocks_on_stderr() -> None:
    spec = build_encoder_spec("libx265", 17)
    cmd = build_chunk_writer_cmd(
        Path("out/chunk_0000.mp4"), 2792, 1816, 29.97,
        spec, "scale=3840:2160:flags=lanczos,setsar=1",
    )
    assert cmd[0] == "ffmpeg"
    # quiet stderr: progress output would otherwise fill the pipe and hang the worker
    assert "-nostats" in cmd and cmd[cmd.index("-loglevel") + 1] == "error"
    assert cmd[cmd.index("-s") + 1] == "2792x1816"
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("scale=3840:2160:flags=lanczos,setsar=1,")
    # RGB -> YUV must use the same matrix the file is tagged with
    assert "out_color_matrix=bt709" in vf and vf.endswith("format=yuv420p")
    assert cmd[cmd.index("-colorspace") + 1] == "bt709"
    assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
    assert cmd[-1] == str(Path("out/chunk_0000.mp4"))


def test_chunk_writer_cmd_x264_has_no_hvc1_tag() -> None:
    cmd = build_chunk_writer_cmd(
        Path("c.mp4"), 640, 480, 25.0, build_encoder_spec("libx264", 17), None
    )
    assert "-tag:v" not in cmd
    assert cmd[cmd.index("-vf") + 1].startswith("scale=out_color_matrix=bt709")


def test_legacy_writer_cmd_unchanged_semantics() -> None:
    cmd = build_legacy_chunk_writer_cmd(Path("c.mp4"), 2880, 1920, 29.97)
    assert cmd[cmd.index("-crf") + 1] == "16"
    assert "-vf" not in cmd
