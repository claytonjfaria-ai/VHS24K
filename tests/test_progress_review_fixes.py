"""Regression tests for the three issues found in review of the progress feature."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vhs24k import progress, upscale
from vhs24k.config import PipelineConfig
from vhs24k.encoding import build_encoder_spec
from vhs24k.preprocess import _run_ffmpeg_with_progress
from vhs24k.progress import (
    RunTracker,
    UpscaleSnapshot,
    chunk_progress_path,
    write_chunk_progress,
)
from vhs24k.upscale import upscale_manifest


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _tracker(tmp_path: Path, sizes: dict[str, int]) -> tuple[RunTracker, io.StringIO]:
    out = io.StringIO()
    return RunTracker(tmp_path, sizes, stream=out, clock=FakeClock()), out


def test_overall_percent_never_goes_backwards_across_sources_in_stage_only_commands(tmp_path: Path) -> None:
    tracker, out = _tracker(tmp_path, {"A": 1000, "B": 3000})
    tracker.begin_source("A")
    tracker.update_upscale(UpscaleSnapshot(100, 100.0, 1, 1, (), None, True), force=True)
    before = tracker.overall_fraction()
    tracker.finish_source("upscale")
    tracker.begin_source("B")
    assert tracker.overall_fraction() >= before - 1e-9
    tracker.finish_source("upscale")
    tracker.close("Upscale finished")
    last = out.getvalue().splitlines()[-1]
    assert "Upscale finished" in last and "Overall 98.0%" in last  # prepare + upscale = 98% of the work


def test_finish_source_rejects_unknown_stage(tmp_path: Path) -> None:
    tracker, _ = _tracker(tmp_path, {"A": 1})
    with pytest.raises(ValueError):
        tracker.finish_source("polish")


def test_failing_progress_callback_kills_ffmpeg_instead_of_hanging() -> None:
    script = "import time; print('out_time_us=1000000', flush=True); time.sleep(60)"

    def boom(_fraction: float) -> None:
        raise ValueError("stdout closed")

    start = time.monotonic()
    with pytest.raises(ValueError, match="stdout closed"):
        _run_ffmpeg_with_progress([sys.executable, "-c", script], 10.0, boom)
    assert time.monotonic() - start < 30, "FFmpeg was left running"


def test_closed_output_stream_never_breaks_reporting(tmp_path: Path) -> None:
    stream = io.StringIO()
    tracker = RunTracker(tmp_path, {"A": 10}, stream=stream, clock=FakeClock())
    stream.close()
    tracker.begin_source("A")  # print() raises ValueError internally; it must be swallowed
    assert (tmp_path / progress.LOG_NAME).exists() and (tmp_path / progress.JSON_NAME).exists()


def _manifest_for(tmp_path: Path) -> tuple[Path, list[dict], PipelineConfig, Path]:
    src = tmp_path / "work" / "VTS_01"
    (src / "chunks").mkdir(parents=True)
    (src / "upscaled").mkdir(parents=True)
    chunks = []
    for i, size in enumerate((100, 300)):
        inp = src / "chunks" / f"chunk_{i:04d}.mp4"
        inp.write_bytes(b"x" * size)
        chunks.append(
            {"index": i, "input_path": str(inp), "output_path": str(src / "upscaled" / inp.name), "status": "pending"}
        )
    manifest = src / "manifest.json"
    manifest.write_text(
        json.dumps({"source_name": "VTS_01", "video_info": {"dar": "4:3"}, "chunks": chunks}), encoding="utf-8"
    )
    esrgan = tmp_path / "Real-ESRGAN"
    esrgan.mkdir()
    cfg = PipelineConfig(
        input_dir=tmp_path / "in", work_dir=tmp_path / "work", output_dir=tmp_path / "out",
        realesrgan_dir=esrgan, num_gpus=1,
    )
    return manifest, chunks, cfg, src / "upscaled"


def test_stale_progress_files_are_cleared_before_workers_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, chunks, cfg, out_dir = _manifest_for(tmp_path)
    for chunk in chunks:  # left over from an interrupted run
        write_chunk_progress(chunk_progress_path(Path(chunk["input_path"]), out_dir), 9, 10)
    monkeypatch.setattr(upscale, "detect_gpus", lambda: [0])
    monkeypatch.setattr(upscale, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17))
    existed_at_launch: list[bool] = []

    def fake_run(cmd, **kwargs):
        existed_at_launch.append(Path(cmd[cmd.index("--progress-file") + 1]).exists())
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(upscale.subprocess, "run", fake_run)
    upscale_manifest(manifest, cfg)
    assert existed_at_launch == [False, False]


def test_failed_chunk_leaves_no_progress_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, cfg, out_dir = _manifest_for(tmp_path)
    monkeypatch.setattr(upscale, "detect_gpus", lambda: [0])
    monkeypatch.setattr(upscale, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17))

    def failing_run(cmd, **kwargs):
        write_chunk_progress(Path(cmd[cmd.index("--progress-file") + 1]), 7, 10)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="CUDA out of memory")

    monkeypatch.setattr(upscale.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="Upscaling failed"):
        upscale_manifest(manifest, cfg)
    assert not list(out_dir.glob("*.progress.json"))
