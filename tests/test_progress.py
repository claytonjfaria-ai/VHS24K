"""File-size-weighted progress: math, logging, the status command, and the worker hooks."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vhs24k import cli, progress, upscale
from vhs24k.config import PipelineConfig
from vhs24k.encoding import build_encoder_spec
from vhs24k.frame_pipeline import run_pipeline
from vhs24k.preprocess import _run_ffmpeg_with_progress
from vhs24k.progress import (
    InFlightChunk,
    RunTracker,
    UpscaleSnapshot,
    chunk_progress_path,
    describe_work_dir,
    format_bytes,
    format_duration,
    read_chunk_progress,
    snapshot_chunks,
    write_chunk_progress,
)
from vhs24k.upscale import upscale_manifest


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _chunks(tmp_path: Path, sizes: list[int], done: int = 0) -> list[dict]:
    """Chunk entries with real input files of the given sizes; the first ``done`` are finished."""
    chunk_dir, out_dir = tmp_path / "chunks", tmp_path / "upscaled"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, size in enumerate(sizes):
        inp = chunk_dir / f"chunk_{i:04d}.mp4"
        inp.write_bytes(b"x" * size)
        out = out_dir / f"chunk_{i:04d}.mp4"
        entries.append(
            {"index": i, "input_path": str(inp), "output_path": str(out),
             "status": "done" if i < done else "pending"}
        )
        if i < done:
            out.write_bytes(b"v")
    return entries


def test_formatters() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(1536) == "1.50 KB"
    assert format_bytes(3 * 1024**3) == "3.00 GB"
    assert format_duration(0) == "0:00:00"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(-5) == "0:00:00"


class TestChunkProgressFiles:
    def test_write_then_read_round_trip(self, tmp_path: Path) -> None:
        path = chunk_progress_path(Path("chunks/chunk_0007.mp4"), tmp_path)
        assert path == tmp_path / "chunk_0007.progress.json"
        write_chunk_progress(path, 450, 1800)
        assert read_chunk_progress(path) == {"frames_done": 450, "total_frames": 1800}

    @pytest.mark.parametrize("content", ["", "not json", "[1, 2]", "null"])
    def test_garbage_or_missing_files_are_ignored(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / "p.progress.json"
        assert read_chunk_progress(path) is None
        path.write_text(content, encoding="utf-8")
        assert read_chunk_progress(path) is None

    def test_writing_never_raises(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        write_chunk_progress(blocker / "child.json", 1, 2)  # parent is a file -> OSError, swallowed
        write_chunk_progress(None, 1, 2)


class TestSnapshot:
    def test_progress_is_weighted_by_file_size_including_partial_chunks(self, tmp_path: Path) -> None:
        chunks = _chunks(tmp_path, [100, 300, 600], done=1)
        write_chunk_progress(chunk_progress_path(Path(chunks[1]["input_path"]), tmp_path / "upscaled"), 900, 1800)
        snap = snapshot_chunks(chunks)
        # 100 (done) + 300 * 50% = 250 of 1000 bytes
        assert snap.total_bytes == 1000 and snap.bytes_known
        assert snap.done_bytes == pytest.approx(250)
        assert snap.fraction == pytest.approx(0.25)
        assert (snap.chunks_done, snap.chunks_total) == (1, 3)
        assert snap.in_flight == (InFlightChunk(1, 0.5),)

    def test_big_chunks_count_for_more_than_small_ones(self, tmp_path: Path) -> None:
        small_done = snapshot_chunks(_chunks(tmp_path / "a", [100, 900], done=1))
        big_done = snapshot_chunks(_chunks(tmp_path / "b", [900, 100], done=1))
        assert small_done.fraction == pytest.approx(0.1)
        assert big_done.fraction == pytest.approx(0.9)

    def test_an_unfinished_chunk_never_reports_100_percent(self, tmp_path: Path) -> None:
        chunks = _chunks(tmp_path, [100])
        write_chunk_progress(chunk_progress_path(Path(chunks[0]["input_path"]), tmp_path / "upscaled"), 1800, 1800)
        assert snapshot_chunks(chunks).fraction < 1.0

    def test_missing_input_files_fall_back_to_chunk_count(self, tmp_path: Path) -> None:
        chunks = _chunks(tmp_path, [100, 100, 100, 100], done=1)
        for c in chunks:
            Path(c["input_path"]).unlink()
        snap = snapshot_chunks(chunks)
        assert not snap.bytes_known
        assert snap.fraction == pytest.approx(0.25)

    def test_empty_chunk_list_is_complete_not_a_crash(self) -> None:
        assert snapshot_chunks([]).fraction == 1.0


class TestRunTracker:
    def _tracker(self, tmp_path: Path, sizes: dict[str, int], clock: FakeClock, **kw):
        out = io.StringIO()
        return RunTracker(tmp_path, sizes, stream=out, clock=clock, **kw), out

    def test_overall_percent_is_weighted_by_source_size_and_stage(self, tmp_path: Path) -> None:
        clock = FakeClock()
        tracker, _ = self._tracker(tmp_path, {"VTS_01": 1000, "VTS_02": 3000}, clock)
        tracker.begin_source("VTS_01")
        tracker.update_prepare(1.0, force=True)
        assert tracker.overall_fraction() == pytest.approx(0.25 * 0.05)
        half = UpscaleSnapshot(1000, 500.0, 10, 5, (), None, True)
        tracker.update_upscale(half, force=True)
        # VTS_01 = prepare done (5%) + half of upscale (46.5%) = 51.5% of 1000 of 4000 bytes
        assert tracker.overall_fraction() == pytest.approx(0.25 * 0.515)
        tracker.update_assemble(1.0)
        tracker.finish_source()
        assert tracker.overall_fraction(after_complete=True) == pytest.approx(0.25)
        tracker.begin_source("VTS_02")
        tracker.update_upscale(UpscaleSnapshot(100, 100.0, 1, 1, (), None, True), force=True)
        assert tracker.overall_fraction() == pytest.approx(0.25 + 0.75 * 0.98)

    def test_each_update_is_printed_logged_and_saved_as_json(self, tmp_path: Path) -> None:
        clock = FakeClock()
        tracker, out = self._tracker(tmp_path, {"VTS_02": 3 * 1024**3}, clock)
        tracker.begin_source("VTS_02")
        snap = UpscaleSnapshot(2000, 1000.0, 8, 4, (InFlightChunk(4, 0.5),), clock(), True)
        tracker.update_upscale(snap, force=True)

        line = out.getvalue().strip().splitlines()[-1]
        assert "Overall 51.5%" in line and "of 3.00 GB source" in line
        assert "chunk 4/8 done" in line and "1 active" in line and "elapsed 0:00:00" in line
        assert line == (tmp_path / progress.LOG_NAME).read_text(encoding="utf-8").strip().splitlines()[-1]
        state = json.loads((tmp_path / progress.JSON_NAME).read_text(encoding="utf-8"))
        assert state["overall_percent"] == pytest.approx(51.5, abs=0.01)
        assert (state["chunks_done"], state["chunks_total"]) == (4, 8)
        assert state["in_flight"] == [{"chunk": 4, "percent": 50.0}]
        assert state["updated_at"] and state["stage"] == "upscale"

    def test_updates_are_rate_limited_unless_forced(self, tmp_path: Path) -> None:
        clock = FakeClock()
        tracker, out = self._tracker(tmp_path, {"A": 100}, clock, min_emit_interval=5.0)
        tracker.begin_source("A")
        before = len(out.getvalue().splitlines())
        snap = UpscaleSnapshot(10, 1.0, 2, 0, (), None, True)
        tracker.update_upscale(snap)
        clock.advance(2)
        tracker.update_upscale(snap)
        assert len(out.getvalue().splitlines()) == before  # both suppressed
        clock.advance(4)
        tracker.update_upscale(snap)
        assert len(out.getvalue().splitlines()) == before + 1
        tracker.update_upscale(snap, force=True)
        assert len(out.getvalue().splitlines()) == before + 2

    def test_eta_appears_once_there_is_a_measured_rate(self, tmp_path: Path) -> None:
        clock = FakeClock()
        tracker, out = self._tracker(tmp_path, {"A": 1000}, clock)
        tracker.begin_source("A")
        clock.advance(100)
        tracker.update_upscale(UpscaleSnapshot(1000, 300.0, 10, 3, (), None, True), force=True)
        assert "ETA" in out.getvalue().splitlines()[-1]
        state = json.loads((tmp_path / progress.JSON_NAME).read_text(encoding="utf-8"))
        assert state["eta_seconds"] is not None and state["eta_seconds"] > 0

    def test_stalled_run_gets_a_warning(self, tmp_path: Path) -> None:
        clock = FakeClock()
        tracker, out = self._tracker(tmp_path, {"A": 1000}, clock)
        tracker.begin_source("A")
        live = UpscaleSnapshot(1000, 100.0, 10, 1, (InFlightChunk(1, 0.1),), clock(), True)
        tracker.update_upscale(live, force=True)
        assert "WARNING" not in out.getvalue()
        clock.advance(progress.STALL_WARNING_SECONDS + 60)
        tracker.update_upscale(live, force=True)  # same stale activity timestamp
        assert "WARNING: no progress for" in out.getvalue().splitlines()[-1]

    def test_test_mode_hides_source_byte_counts(self, tmp_path: Path) -> None:
        clock = FakeClock()
        tracker, out = self._tracker(tmp_path, {"VTS_02": 1}, clock, show_source_bytes=False)
        tracker.begin_source("VTS_02")
        tracker.update_upscale(UpscaleSnapshot(10, 5.0, 1, 0, (), None, True), force=True)
        assert "source)" not in out.getvalue() and "Overall 51.5%" in out.getvalue()

    def test_close_reports_done(self, tmp_path: Path) -> None:
        tracker, out = self._tracker(tmp_path, {"A": 10}, FakeClock())
        tracker.begin_source("A")
        tracker.finish_source()
        tracker.close()
        assert "Overall 100.0%" in out.getvalue().splitlines()[-1] and "All done" in out.getvalue()


class TestStatusCommand:
    def _work_dir(self, tmp_path: Path) -> tuple[Path, Path]:
        work = tmp_path / "work"
        src = work / "VTS_02"
        chunks = _chunks(src, [100, 300, 600], done=1)
        write_chunk_progress(
            chunk_progress_path(Path(chunks[1]["input_path"]), src / "upscaled"), 900, 1800
        )
        manifest = src / "manifest.json"
        manifest.write_text(json.dumps({"source_name": "VTS_02", "chunks": chunks}), encoding="utf-8")
        return work, manifest

    def test_status_lines_show_percent_chunks_and_active_work(self, tmp_path: Path) -> None:
        work, manifest = self._work_dir(tmp_path)
        text = "\n".join(describe_work_dir(work, [manifest]))
        assert "VTS_02: upscale 25.0% (1/3 chunks" in text
        assert "chunk 1 50%" in text
        assert "No progress.json yet" in text

    def test_status_warns_when_the_log_has_gone_quiet(self, tmp_path: Path) -> None:
        work, manifest = self._work_dir(tmp_path)
        (work / progress.JSON_NAME).write_text(
            json.dumps({"updated_at": "2026-01-01T00:00:00", "overall_percent": 40, "stage": "upscale"}),
            encoding="utf-8",
        )
        later = os.path.getmtime(work / progress.JSON_NAME) + 7200
        text = "\n".join(describe_work_dir(work, [manifest], now=later))
        assert "2:00:00 ago" in text and "may have stopped or hung" in text

    def test_cli_status_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        work, _ = self._work_dir(tmp_path)
        assert cli.main(["status", "--work-dir", str(work)]) == 0
        assert "upscale 25.0%" in capsys.readouterr().out

    def test_cli_status_without_a_work_dir_is_a_clear_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["status", "--work-dir", str(tmp_path / "nothing")]) == 1
        assert "Nothing has been prepared" in capsys.readouterr().out


class TestHooks:
    def test_pipeline_reports_every_frame_in_order(self) -> None:
        class Sink:
            def write(self, data: bytes) -> None: ...
            def close(self) -> None: ...
            def abort(self) -> None: ...

        seen: list[int] = []
        count = run_pipeline(iter(range(40)), lambda x: x, lambda x: bytes([x]), Sink(), on_frame=seen.append)
        assert count == 40 and seen == list(range(1, 41))

    def test_prepare_progress_is_parsed_from_ffmpeg_progress_output(self) -> None:
        script = (
            "import sys\n"
            "for us in ('N/A', 5000000, 'bogus', 10000000, 20000000, 99000000):\n"
            "    print(f'out_time_us={us}'); print('progress=continue')\n"
            "sys.stdout.flush()\n"
        )
        fractions: list[float] = []
        _run_ffmpeg_with_progress([sys.executable, "-c", script], 20.0, fractions.append)
        assert fractions == [0.25, 0.5, 1.0, 1.0]  # unparseable lines skipped, never above 100%

    def test_prepare_failure_surfaces_ffmpeg_error(self) -> None:
        script = "import sys; sys.stderr.write('moov atom not found'); sys.exit(1)"
        with pytest.raises(RuntimeError, match="moov atom not found"):
            _run_ffmpeg_with_progress([sys.executable, "-c", script], 10.0, lambda f: None)

    def test_worker_command_gets_a_progress_file(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(input_dir=tmp_path, work_dir=tmp_path, output_dir=tmp_path)
        cmd = upscale.build_enhancer_worker_cmd(
            Path("i.mp4"), Path("o.mp4"), cfg, Path("R"), progress_file=tmp_path / "p.json"
        )
        assert cmd[cmd.index("--progress-file") + 1] == str(tmp_path / "p.json")
        plain = upscale.build_enhancer_worker_cmd(Path("i"), Path("o"), cfg, Path("R"))
        assert "--progress-file" not in plain


def test_upscale_manifest_reports_progress_and_cleans_up_progress_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "work" / "VTS_01"
    chunks = _chunks(src, [100, 300])
    audio = src / "audio.mka"
    audio.write_bytes(b"a")
    manifest_path = src / "manifest.json"
    manifest_path.write_text(
        json.dumps({"source_name": "VTS_01", "video_info": {"dar": "4:3"}, "audio_path": str(audio), "chunks": chunks}),
        encoding="utf-8",
    )
    esrgan = tmp_path / "Real-ESRGAN"
    esrgan.mkdir()
    cfg = PipelineConfig(
        input_dir=tmp_path / "in", work_dir=tmp_path / "work", output_dir=tmp_path / "out",
        realesrgan_dir=esrgan, num_gpus=1,
    )
    monkeypatch.setattr(upscale, "detect_gpus", lambda: [0])
    monkeypatch.setattr(upscale, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17))
    progress_files: list[Path] = []

    def fake_run(cmd, **kwargs):
        # the real worker writes its progress file; make sure the parent told it where
        pf = Path(cmd[cmd.index("--progress-file") + 1])
        write_chunk_progress(pf, 5, 10)
        progress_files.append(pf)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(upscale.subprocess, "run", fake_run)
    out = io.StringIO()
    tracker = RunTracker(tmp_path / "work", {"VTS_01": 400}, stream=out)
    tracker.begin_source("VTS_01")
    upscale_manifest(manifest_path, cfg, tracker)

    assert len(progress_files) == 2 and not any(p.exists() for p in progress_files)
    state = json.loads((tmp_path / "work" / progress.JSON_NAME).read_text(encoding="utf-8"))
    assert state["chunks_done"] == 2 and state["chunks_total"] == 2
    assert "chunk 2/2 done" in out.getvalue().splitlines()[-1]
    assert (tmp_path / "work" / progress.LOG_NAME).read_text(encoding="utf-8").count("[Progress") >= 2


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None or __import__("shutil").which("ffprobe") is None,
    reason="ffmpeg not installed",
)
def test_real_preprocessing_reports_progress_with_real_ffmpeg(tmp_path: Path) -> None:
    """Guards the exact FFmpeg flags (-progress pipe:1 -nostats) against a real FFmpeg build."""
    from vhs24k.preprocess import SourceGroup, prepare_chunks

    video = tmp_path / "in" / "Wedding1.mp4"
    video.parent.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=25:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    cfg = PipelineConfig(
        input_dir=video.parent, work_dir=tmp_path / "work", output_dir=tmp_path / "out", chunk_seconds=2
    )
    out = io.StringIO()
    tracker = RunTracker(tmp_path / "work", {"Wedding1": video.stat().st_size}, stream=out)
    tracker.begin_source("Wedding1")
    manifest = json.loads(
        prepare_chunks(SourceGroup("Wedding1", [video], "mp4"), cfg.work_dir, cfg, tracker).read_text(encoding="utf-8")
    )
    assert len(manifest["chunks"]) >= 1  # chunks only split at keyframes, so a short clip may be one
    lines = out.getvalue().splitlines()
    assert any("preparing chunks" in line for line in lines)
    assert "Overall 5.0%" in lines[-1]  # prepare finished = its full 5% share of the source
    # a resumed run that finds the work already prepared still reports prepare as done
    out.truncate(0)
    out.seek(0)
    prepare_chunks(SourceGroup("Wedding1", [video], "mp4"), cfg.work_dir, cfg, tracker)
    assert "Overall 5.0%" in out.getvalue()
