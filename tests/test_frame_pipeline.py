"""The threaded frame pipeline: ordering, error propagation, and no silent hangs."""

from __future__ import annotations

import shutil
import sys
import threading

import pytest

from vhs24k.frame_pipeline import FfmpegSink, PipelineError, run_pipeline


class FakeSink:
    def __init__(self, fail_on: int | None = None) -> None:
        self.frames: list[bytes] = []
        self.closed = False
        self.aborted = False
        self._fail_on = fail_on

    def write(self, data: bytes) -> None:
        if self._fail_on is not None and len(self.frames) >= self._fail_on:
            raise PipelineError("sink full")
        self.frames.append(data)

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


def _run(frames, enhance=lambda x: x, to_bytes=lambda x: bytes([x]), sink=None, depth=3):
    sink = sink or FakeSink()
    done = {}

    def target() -> None:
        try:
            done["count"] = run_pipeline(frames, enhance, to_bytes, sink, queue_depth=depth)
        except BaseException as exc:  # noqa: BLE001
            done["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert not thread.is_alive(), "pipeline hung"
    return sink, done


def test_frames_are_processed_in_order() -> None:
    sink, done = _run(iter(range(100)), enhance=lambda x: x + 1, to_bytes=lambda x: bytes([x]))
    assert done["count"] == 100
    assert [f[0] for f in sink.frames] == [i + 1 for i in range(100)]
    assert sink.closed and not sink.aborted


def test_empty_input_finishes_cleanly() -> None:
    sink, done = _run(iter(()))
    assert done["count"] == 0 and sink.closed


def test_error_in_prepare_stage_propagates_and_aborts_sink() -> None:
    def frames():
        yield 1
        yield 2
        raise ValueError("bad frame")

    sink, done = _run(frames())
    assert isinstance(done["error"], PipelineError) and "bad frame" in str(done["error"])
    assert sink.aborted and not sink.closed


def test_error_in_gpu_stage_propagates() -> None:
    def enhance(x: int) -> int:
        if x == 5:
            raise RuntimeError("CUDA out of memory")
        return x

    sink, done = _run(iter(range(50)), enhance=enhance)
    assert "CUDA out of memory" in str(done["error"])
    assert sink.aborted


def test_error_in_writer_stage_propagates() -> None:
    sink, done = _run(iter(range(50)), sink=FakeSink(fail_on=3))
    assert "sink full" in str(done["error"])
    assert sink.aborted


def test_prepare_generator_is_closed_on_failure() -> None:
    closed = threading.Event()

    def frames():
        try:
            for i in range(1000):
                yield i
        finally:
            closed.set()

    _run(frames(), enhance=lambda x: (_ for _ in ()).throw(RuntimeError("boom")))
    assert closed.wait(timeout=5), "capture would leak"


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_sink_drains_stderr_so_chatty_encoders_cannot_deadlock() -> None:
    """A child writing far more than a pipe buffer to stderr must not block the writer."""
    script = (
        "import sys\n"
        "sys.stderr.write('x' * 2_000_000)\n"
        "sys.stderr.flush()\n"
        "sys.stdin.buffer.read()\n"
    )
    sink = FfmpegSink([sys.executable, "-c", script])
    sink.write(b"frame")
    sink.close()  # would hang forever without a stderr drain


def test_sink_reports_encoder_failure_with_stderr_tail() -> None:
    script = "import sys; sys.stderr.write('encoder exploded'); sys.exit(3)"
    sink = FfmpegSink([sys.executable, "-c", script])
    with pytest.raises(PipelineError, match="encoder exploded"):
        sink.close()


@needs_ffmpeg
def test_real_ffmpeg_sink_encodes_frames(tmp_path) -> None:
    out = tmp_path / "out.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48", "-r", "25", "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ]
    frame = bytes(64 * 48 * 3)
    count = run_pipeline(iter([frame] * 10), lambda f: f, lambda f: f, FfmpegSink(cmd))
    assert count == 10 and out.stat().st_size > 0
