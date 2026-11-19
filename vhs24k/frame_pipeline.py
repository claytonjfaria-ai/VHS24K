"""Threaded frame pipeline: CPU prepare -> GPU enhance -> FFmpeg encode, all overlapped.

The original worker did these steps strictly one after another per frame, so the
GPU sat idle during CPU work and vice versa. Here each stage runs in its own
thread (OpenCV, NumPy, PyTorch and the FFmpeg pipe all release the GIL), so
throughput is set by the slowest stage rather than the sum of all stages.
"""

from __future__ import annotations

import collections
import logging
import queue
import subprocess
import threading
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

_QUEUE_POLL_SECONDS = 0.25
_STDERR_TAIL_BYTES = 4000
_END = object()


class PipelineError(RuntimeError):
    """Raised when any pipeline stage or the FFmpeg writer fails."""


class FfmpegSink:
    """FFmpeg process fed with raw frames on stdin; stderr is drained continuously.

    Draining matters: if FFmpeg's stderr pipe fills up while nobody reads it,
    FFmpeg blocks, the stdin pipe stops being consumed, and the worker hangs forever.
    """

    def __init__(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if self._proc.stdin is None or self._proc.stderr is None:
            raise PipelineError("Failed to open FFmpeg pipes.")
        self._stderr_tail: collections.deque[bytes] = collections.deque()
        self._tail_size = 0
        self._drain = threading.Thread(target=self._drain_stderr, name="ffmpeg-stderr", daemon=True)
        self._drain.start()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for chunk in iter(lambda: self._proc.stderr.read(4096), b""):  # type: ignore[union-attr]
            self._stderr_tail.append(chunk)
            self._tail_size += len(chunk)
            while self._tail_size > _STDERR_TAIL_BYTES and len(self._stderr_tail) > 1:
                self._tail_size -= len(self._stderr_tail.popleft())

    def stderr_tail(self) -> str:
        return b"".join(self._stderr_tail).decode("utf-8", errors="replace")[-_STDERR_TAIL_BYTES:]

    def write(self, data: bytes) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(data)
        except (BrokenPipeError, OSError) as exc:
            raise PipelineError(f"FFmpeg writer closed unexpectedly: {self.stderr_tail()}") from exc

    def close(self) -> None:
        """Finish the stream and raise if FFmpeg reported a failure."""
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        rc = self._proc.wait()
        self._drain.join(timeout=5)
        if rc != 0:
            raise PipelineError(f"FFmpeg writer failed (exit {rc}): {self.stderr_tail()}")

    def abort(self) -> None:
        """Kill FFmpeg without raising (used on the error path)."""
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._drain.join(timeout=5)


def _put(q: queue.Queue[Any], item: Any, stop: threading.Event) -> bool:
    """Put ``item`` unless the pipeline is stopping; returns False when stopped."""
    while not stop.is_set():
        try:
            q.put(item, timeout=_QUEUE_POLL_SECONDS)
            return True
        except queue.Full:
            continue
    return False


def _get(q: queue.Queue[Any], stop: threading.Event) -> Any:
    """Get an item, returning ``_END`` when the pipeline is stopping."""
    while not stop.is_set():
        try:
            return q.get(timeout=_QUEUE_POLL_SECONDS)
        except queue.Empty:
            continue
    return _END


def run_pipeline(
    prepared_frames: Iterable[Any],
    enhance: Callable[[Any], Any],
    to_bytes: Callable[[Any], bytes],
    sink: FfmpegSink,
    queue_depth: int = 6,
) -> int:
    """Run all stages to completion and return the number of frames written.

    ``prepared_frames`` is consumed in a background thread (CPU stage),
    ``enhance`` plus ``to_bytes`` run on the calling thread (GPU stage), and a
    writer thread feeds the encoder. The first error from any stage stops the rest.
    """
    stop = threading.Event()
    errors: list[BaseException] = []
    q_in: queue.Queue[Any] = queue.Queue(maxsize=queue_depth)
    q_out: queue.Queue[Any] = queue.Queue(maxsize=queue_depth)
    written = [0]

    def fail(exc: BaseException) -> None:
        errors.append(exc)
        stop.set()

    def produce() -> None:
        iterator = iter(prepared_frames)
        try:
            for item in iterator:
                if not _put(q_in, item, stop):
                    return
            _put(q_in, _END, stop)
        except BaseException as exc:  # noqa: BLE001 - reported via fail()
            fail(exc)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def write() -> None:
        try:
            while True:
                item = _get(q_out, stop)
                if item is _END:
                    return
                sink.write(item)
                written[0] += 1
        except BaseException as exc:  # noqa: BLE001
            fail(exc)

    producer = threading.Thread(target=produce, name="frame-prepare", daemon=True)
    writer = threading.Thread(target=write, name="frame-writer", daemon=True)
    producer.start()
    writer.start()

    try:
        while True:
            item = _get(q_in, stop)
            if item is _END:
                break
            if not _put(q_out, to_bytes(enhance(item)), stop):
                break
        _put(q_out, _END, stop)
    except BaseException as exc:  # noqa: BLE001
        fail(exc)

    producer.join()
    writer.join()

    if errors:
        sink.abort()
        raise PipelineError(str(errors[0])) from errors[0]
    sink.close()
    return written[0]
