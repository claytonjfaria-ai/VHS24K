"""File-size-weighted progress tracking, logging, and status reporting.

Progress is measured in bytes so large and small pieces count in proportion:

* **Overall** percentage is weighted by the size of the original source files
  (the VOBs), split into stages: prepare, upscale, assemble.
* **Upscale** percentage is weighted by the size of each prepared chunk file.
  Chunks still being processed count partially, from a small progress file the
  worker rewrites every few frames.

Each update is printed, appended to ``<work_dir>/progress.log`` and written to
``<work_dir>/progress.json`` (latest snapshot, includes ``updated_at`` so a
frozen run is easy to spot).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

# Share of a source's progress assigned to each stage (sums to 1.0).
STAGE_WEIGHTS: dict[str, float] = {"prepare": 0.05, "upscale": 0.93, "assemble": 0.02}
LOG_NAME = "progress.log"
JSON_NAME = "progress.json"
DEFAULT_EMIT_INTERVAL_SECONDS = 5.0
MONITOR_INTERVAL_SECONDS = 15.0
STALL_WARNING_SECONDS = 900.0
FRAME_REPORT_INTERVAL = 15
MIN_ETA_ELAPSED_SECONDS = 30.0
MAX_IN_FLIGHT_FRACTION = 0.99


def format_bytes(num_bytes: float) -> str:
    """Human-readable size, e.g. ``1.28 GB``."""
    value = float(max(0.0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"


def format_duration(seconds: float) -> str:
    """``H:MM:SS`` for a non-negative number of seconds."""
    total = int(max(0.0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------- per-chunk progress files
def chunk_progress_path(chunk_input: Path, output_dir: Path) -> Path:
    """Location of the small progress file the worker writes for one chunk."""
    return Path(output_dir) / f"{Path(chunk_input).stem}.progress.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_chunk_progress(path: Path | None, frames_done: int, total_frames: int) -> None:
    """Record worker progress. Never raises: progress reporting must not fail a chunk."""
    if path is None:
        return
    try:
        _atomic_write_text(
            Path(path),
            json.dumps({"frames_done": int(frames_done), "total_frames": int(total_frames)}),
        )
    except OSError:
        pass


def read_chunk_progress(path: Path) -> dict[str, Any] | None:
    """Read a worker progress file, or ``None`` if missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- manifest snapshot
@dataclass(frozen=True)
class InFlightChunk:
    index: int
    fraction: float


@dataclass(frozen=True)
class UpscaleSnapshot:
    """Size-weighted state of a source's chunks at one moment."""

    total_bytes: int
    done_bytes: float
    chunks_total: int
    chunks_done: int
    in_flight: tuple[InFlightChunk, ...]
    last_activity: float | None  # epoch seconds of the newest chunk progress/output
    bytes_known: bool

    @property
    def fraction(self) -> float:
        return 1.0 if self.total_bytes <= 0 else min(1.0, self.done_bytes / self.total_bytes)


def _size_of(path_str: str | None) -> int:
    try:
        return os.path.getsize(path_str) if path_str else 0
    except OSError:
        return 0


def snapshot_chunks(chunks: list[dict[str, Any]]) -> UpscaleSnapshot:
    """Compute size-weighted progress for a manifest's chunk list."""
    sizes = [_size_of(c.get("input_path")) for c in chunks]
    bytes_known = bool(sizes) and all(s > 0 for s in sizes)
    weights = sizes if bytes_known else [1] * len(chunks)
    total = sum(weights)

    done_bytes = 0.0
    chunks_done = 0
    in_flight: list[InFlightChunk] = []
    last_activity: float | None = None

    for chunk, weight in zip(chunks, weights):
        if chunk.get("status") == "done":
            done_bytes += weight
            chunks_done += 1
            out = chunk.get("output_path")
            if out and os.path.exists(out):
                last_activity = max(last_activity or 0.0, os.path.getmtime(out))
            continue
        input_path, out_path = chunk.get("input_path"), chunk.get("output_path")
        if not input_path or not out_path:
            continue
        prog_path = chunk_progress_path(Path(input_path), Path(out_path).parent)
        data = read_chunk_progress(prog_path)
        if not data:
            continue
        total_frames = int(data.get("total_frames") or 0)
        frames_done = int(data.get("frames_done") or 0)
        fraction = min(MAX_IN_FLIGHT_FRACTION, frames_done / total_frames) if total_frames > 0 else 0.0
        done_bytes += weight * fraction
        in_flight.append(InFlightChunk(int(chunk.get("index", 0)), fraction))
        try:
            last_activity = max(last_activity or 0.0, prog_path.stat().st_mtime)
        except OSError:
            pass

    return UpscaleSnapshot(
        total_bytes=total,
        done_bytes=done_bytes,
        chunks_total=len(chunks),
        chunks_done=chunks_done,
        in_flight=tuple(in_flight),
        last_activity=last_activity,
        bytes_known=bytes_known,
    )


# ---------------------------------------------------------------- run tracker
class RunTracker:
    """Combines stage progress into one overall, size-weighted percentage and logs it."""

    def __init__(
        self,
        work_dir: Path,
        source_bytes: dict[str, int],
        *,
        show_source_bytes: bool = True,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.time,
        min_emit_interval: float = DEFAULT_EMIT_INTERVAL_SECONDS,
    ) -> None:
        self._work_dir = Path(work_dir)
        self._source_bytes = {name: max(0, int(b)) for name, b in source_bytes.items()}
        self._show_source_bytes = show_source_bytes
        self._stream = stream
        self._clock = clock
        self._min_emit_interval = min_emit_interval
        self._start = clock()
        self._completed_bytes = 0.0
        self._source: str | None = None
        self._stage = "prepare"
        self._stage_fraction = 0.0
        self._detail = ""
        self._snapshot: UpscaleSnapshot | None = None
        self._last_emit = float("-inf")
        self._baseline_bytes: float | None = None
        self._baseline_time = self._start
        self._seen_chunks_done = 0
        self._last_activity = self._start

    # ----- state updates -------------------------------------------------
    def begin_source(self, name: str) -> None:
        self._source = name
        self._stage, self._stage_fraction, self._detail, self._snapshot = "prepare", 0.0, "", None
        self._seen_chunks_done = 0
        self._last_activity = self._clock()
        self._emit(force=True, headline=f"Starting {name}")

    def update_prepare(self, fraction: float, force: bool = False) -> None:
        self._stage, self._stage_fraction = "prepare", min(1.0, max(0.0, fraction))
        self._detail = "preparing chunks (estimated from video time processed)"
        self._emit(force=force)

    def update_upscale(self, snapshot: UpscaleSnapshot, force: bool = False) -> None:
        self._stage, self._stage_fraction, self._snapshot = "upscale", snapshot.fraction, snapshot
        if snapshot.chunks_done > self._seen_chunks_done or snapshot.in_flight:
            self._last_activity = max(self._last_activity, snapshot.last_activity or self._clock())
        self._seen_chunks_done = snapshot.chunks_done
        self._detail = self._upscale_detail(snapshot)
        self._emit(force=force)

    def update_assemble(self, fraction: float, force: bool = True) -> None:
        self._stage, self._stage_fraction = "assemble", min(1.0, max(0.0, fraction))
        self._detail = "joining chunks and adding audio" if fraction < 1.0 else "assembly finished"
        self._emit(force=force)

    def finish_source(self, through_stage: str = "assemble") -> None:
        """Credit the current source up to ``through_stage`` (default: fully complete).

        Stage-only commands (``prepare``, ``upscale``) pass their own stage so the overall
        percentage keeps rising across several sources instead of jumping back.
        """
        if through_stage not in STAGE_WEIGHTS:
            raise ValueError(f"Unknown stage: {through_stage!r}")
        if self._source is not None:
            credited = 0.0
            for stage, weight in STAGE_WEIGHTS.items():
                credited += weight
                if stage == through_stage:
                    break
            self._completed_bytes += self._source_bytes.get(self._source, 0) * credited
        self._stage, self._stage_fraction = through_stage, 1.0
        self._emit(force=True, headline=f"Finished {self._source} ({through_stage})", after_complete=True)
        self._source = None

    def close(self, headline: str = "All done") -> None:
        self._emit(force=True, headline=headline, after_complete=True)

    # ----- computation -----------------------------------------------------
    def _source_fraction(self) -> float:
        done = 0.0
        for stage, weight in STAGE_WEIGHTS.items():
            if stage == self._stage:
                done += weight * self._stage_fraction
                break
            done += weight
        return min(1.0, done)

    def _overall_bytes(self, after_complete: bool) -> tuple[float, float]:
        total = float(sum(self._source_bytes.values()))
        if after_complete or self._source is None:
            return self._completed_bytes, total
        current = self._source_bytes.get(self._source, 0)
        return self._completed_bytes + current * self._source_fraction(), total

    def overall_fraction(self, after_complete: bool = False) -> float:
        done, total = self._overall_bytes(after_complete)
        if total > 0:
            return min(1.0, done / total)
        return min(1.0, self._source_fraction())

    def _upscale_detail(self, snap: UpscaleSnapshot) -> str:
        unit = (
            f"{format_bytes(snap.done_bytes)} of {format_bytes(snap.total_bytes)} chunk data"
            if snap.bytes_known
            else "weighted by chunk count"
        )
        active = f"; {len(snap.in_flight)} active" if snap.in_flight else ""
        return (
            f"upscale {snap.fraction * 100:.1f}% (chunk {snap.chunks_done}/{snap.chunks_total} done, "
            f"{unit}{active})"
        )

    def _eta_seconds(self, overall_done: float, overall_total: float, now: float) -> float | None:
        if overall_total <= 0:
            return None
        if self._baseline_bytes is None:
            self._baseline_bytes, self._baseline_time = overall_done, now
            return None
        elapsed = now - self._baseline_time
        gained = overall_done - self._baseline_bytes
        if elapsed < MIN_ETA_ELAPSED_SECONDS or gained <= 0:
            return None
        return (overall_total - overall_done) / (gained / elapsed)

    # ----- output ------------------------------------------------------------
    def _emit(self, force: bool = False, headline: str = "", after_complete: bool = False) -> None:
        now = self._clock()
        if not force and now - self._last_emit < self._min_emit_interval:
            return
        self._last_emit = now

        overall_done, overall_total = self._overall_bytes(after_complete)
        fraction = self.overall_fraction(after_complete)
        eta = None if after_complete else self._eta_seconds(overall_done, overall_total, now)
        elapsed = now - self._start

        parts = [f"Overall {fraction * 100:.1f}%"]
        if self._show_source_bytes and overall_total > 0:
            parts[0] += f" ({format_bytes(overall_done)} of {format_bytes(overall_total)} source)"
        label = headline or (f"{self._source} {self._detail}" if self._source else "")
        if label:
            parts.append(label)
        parts.append(f"elapsed {format_duration(elapsed)}")
        if eta is not None:
            parts.append(f"ETA ~{format_duration(eta)}")
        stalled_for = now - self._last_activity
        if self._stage == "upscale" and self._snapshot and self._snapshot.in_flight and stalled_for > STALL_WARNING_SECONDS:
            parts.append(f"WARNING: no progress for {stalled_for / 60:.0f} min")
        line = f"[Progress {datetime.fromtimestamp(now):%H:%M:%S}] " + " | ".join(parts)

        self._write_line(line)
        self._write_json(fraction, overall_done, overall_total, eta, elapsed, now)

    def _write_line(self, line: str) -> None:
        try:
            print(line, file=self._stream or sys.stdout, flush=True)
        except (OSError, ValueError):  # e.g. closed stdout: reporting must never stop the run
            pass
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            with open(self._work_dir / LOG_NAME, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def _write_json(
        self, fraction: float, done: float, total: float, eta: float | None, elapsed: float, now: float
    ) -> None:
        snap = self._snapshot
        payload = {
            "updated_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "source": self._source,
            "stage": self._stage,
            "overall_percent": round(fraction * 100, 2),
            "source_bytes_done": int(done),
            "source_bytes_total": int(total),
            "stage_percent": round(self._stage_fraction * 100, 2),
            "chunks_done": snap.chunks_done if snap else None,
            "chunks_total": snap.chunks_total if snap else None,
            "in_flight": [
                {"chunk": c.index, "percent": round(c.fraction * 100, 1)} for c in (snap.in_flight if snap else ())
            ],
            "elapsed_seconds": int(elapsed),
            "eta_seconds": None if eta is None else int(eta),
        }
        try:
            _atomic_write_text(self._work_dir / JSON_NAME, json.dumps(payload, indent=2))
        except OSError:
            pass


# ---------------------------------------------------------------- status command
def describe_work_dir(work_dir: Path, manifests: list[Path], now: float | None = None) -> list[str]:
    """Plain-text progress report for ``cli status``; reads files only, needs no GPU."""
    now = time.time() if now is None else now
    lines: list[str] = []
    total_bytes = 0.0
    done_bytes = 0.0
    for manifest_path in manifests:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lines.append(f"{Path(manifest_path).parent.name}: manifest unreadable")
            continue
        chunks = manifest.get("chunks", []) if isinstance(manifest, dict) else manifest
        snap = snapshot_chunks(chunks)
        name = manifest.get("source_name", Path(manifest_path).parent.name) if isinstance(manifest, dict) else "?"
        total_bytes += snap.total_bytes if snap.bytes_known else 0
        done_bytes += snap.done_bytes if snap.bytes_known else 0
        active = ", ".join(f"chunk {c.index} {c.fraction * 100:.0f}%" for c in snap.in_flight) or "none running"
        size = (
            f", {format_bytes(snap.done_bytes)} of {format_bytes(snap.total_bytes)}" if snap.bytes_known else ""
        )
        lines.append(
            f"{name}: upscale {snap.fraction * 100:.1f}% ({snap.chunks_done}/{snap.chunks_total} chunks{size}); {active}"
        )
    if total_bytes > 0:
        lines.append(f"All tapes (chunk data): {done_bytes / total_bytes * 100:.1f}%")

    json_path = Path(work_dir) / JSON_NAME
    try:
        info = json.loads(json_path.read_text(encoding="utf-8"))
        age = now - json_path.stat().st_mtime
        lines.append(
            f"Last logged update: {info.get('updated_at')} ({format_duration(age)} ago) - "
            f"overall {info.get('overall_percent')}%, stage {info.get('stage')}"
        )
        if age > STALL_WARNING_SECONDS:
            lines.append("WARNING: no progress update for a long time; the run may have stopped or hung.")
    except (OSError, ValueError):
        lines.append("No progress.json yet (run has not started logging).")
    return lines
