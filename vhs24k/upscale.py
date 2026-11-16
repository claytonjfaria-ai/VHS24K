"""GPU upscaling orchestration for VHS24K using Real-ESRGAN."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from vhs24k.config import PipelineConfig

logger = logging.getLogger(__name__)


def detect_gpus() -> list[int]:
    """Return available CUDA GPU indices via ``nvidia-smi -L`` or PyTorch.

    Falls back to ``[]`` if no GPUs are detected or tools are unavailable.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            indices: list[int] = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("GPU "):
                    prefix = line.split(":", 1)[0]
                    parts = prefix.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        indices.append(int(parts[1]))
            if indices:
                return indices
    except OSError:
        pass

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass

    return []


def build_realesrgan_cmd(
    chunk_input: Path,
    chunk_output: Path,
    cfg: PipelineConfig,
    realesrgan_dir: Path,
) -> list[str]:
    """Build the ``inference_realesrgan_video.py`` command for a video chunk."""
    script_path = realesrgan_dir / "inference_realesrgan_video.py"
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "-i",
        str(chunk_input),
        "-o",
        str(chunk_output),
        "-n",
        cfg.model_name,
        "-s",
        "4",
        "--tile",
        str(cfg.tile_size),
    ]
    if cfg.model_name == "realesr-general-x4v3":
        cmd.extend(["--dn", str(cfg.denoise_strength)])
    if cfg.face_enhance:
        cmd.append("--face_enhance")
    if not cfg.fp16:
        cmd.append("--fp32")
    return cmd


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON payload to ``path`` using a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _resolve_chunk_io(chunk: dict[str, Any], cfg: PipelineConfig) -> tuple[Path, Path]:
    """Extract input file path and Real-ESRGAN output directory from a chunk entry."""
    raw_in = (
        chunk.get("input_path")
        or chunk.get("chunk_path")
        or chunk.get("path")
        or chunk.get("input")
    )
    if not raw_in:
        raise ValueError(f"Chunk entry is missing an input path: {chunk}")
    chunk_input = Path(raw_in)

    raw_out = chunk.get("output_dir") or chunk.get("upscaled_dir")
    if raw_out:
        chunk_output = Path(raw_out)
    elif chunk.get("output_path"):
        out_path = Path(chunk["output_path"])
        chunk_output = out_path.parent if out_path.suffix else out_path
    else:
        chunk_output = Path(cfg.work_dir) / "upscaled"

    return chunk_input, chunk_output


def _expected_realesrgan_output(chunk_input: Path, chunk_output: Path) -> Path:
    """Return the default output video path produced by Real-ESRGAN."""
    return chunk_output / f"{chunk_input.stem}_out{chunk_input.suffix}"


def upscale_manifest(manifest_path: Path, cfg: PipelineConfig) -> Path:
    """Upscale all pending chunks in ``manifest.json`` across available GPUs.

    Updates ``manifest.json`` atomically after each completed chunk so
    interrupted sessions can resume cleanly.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest_data, dict):
        chunks = manifest_data.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError(f"Manifest dict missing 'chunks' list: {manifest_path}")
    elif isinstance(manifest_data, list):
        chunks = manifest_data
    else:
        raise ValueError(f"Unsupported manifest structure in {manifest_path}")

    pending_indices = [
        idx for idx, chunk in enumerate(chunks) if chunk.get("status") != "done"
    ]
    if not pending_indices:
        logger.info("All %d chunks in %s are already done.", len(chunks), manifest_path)
        return manifest_path

    if cfg.realesrgan_dir is None:
        if not cfg.dry_run:
            raise ValueError(
                "PipelineConfig.realesrgan_dir must be set to the Real-ESRGAN repository path."
            )
        realesrgan_dir = Path("Real-ESRGAN")
    else:
        realesrgan_dir = Path(cfg.realesrgan_dir)
    if not cfg.dry_run and not (realesrgan_dir / "inference_realesrgan_video.py").exists():
        raise FileNotFoundError(
            f"Real-ESRGAN script not found at {realesrgan_dir / 'inference_realesrgan_video.py'}"
        )

    detected_gpus = detect_gpus()
    num_workers = max(1, min(cfg.num_gpus, max(1, len(detected_gpus))))
    gpu_queue: queue.Queue[int | None] = queue.Queue()
    if detected_gpus:
        for gpu_id in detected_gpus[:num_workers]:
            gpu_queue.put(gpu_id)
    else:
        for _ in range(num_workers):
            gpu_queue.put(None)

    lock = threading.Lock()
    failures: list[str] = []

    def _process_chunk(chunk_idx: int) -> None:
        chunk = chunks[chunk_idx]
        chunk_input, chunk_output = _resolve_chunk_io(chunk, cfg)
        chunk_output.mkdir(parents=True, exist_ok=True)
        cmd = build_realesrgan_cmd(chunk_input, chunk_output, cfg, realesrgan_dir)

        gpu_id = gpu_queue.get()
        try:
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            if cfg.dry_run:
                gpu_label = f"GPU {gpu_id}" if gpu_id is not None else "CPU/default"
                logger.info("[dry-run][%s] %s", gpu_label, " ".join(cmd))
                return

            proc = subprocess.run(
                cmd,
                cwd=str(realesrgan_dir),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                err_tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
                with lock:
                    chunk["status"] = "failed"
                    chunk["error"] = f"exit={proc.returncode}: {err_tail}"
                    _atomic_write_json(manifest_path, manifest_data)
                raise RuntimeError(
                    f"Real-ESRGAN failed on {chunk_input.name} (exit {proc.returncode}): {err_tail}"
                )

            produced = _expected_realesrgan_output(chunk_input, chunk_output)
            target_raw = chunk.get("output_path")
            if target_raw and Path(target_raw).suffix and produced.exists():
                target_path = Path(target_raw)
                if produced != target_path:
                    os.replace(produced, target_path)
                    produced = target_path

            with lock:
                chunk["status"] = "done"
                chunk.pop("error", None)
                if produced.exists():
                    chunk["upscaled_path"] = str(produced)
                    chunk["output_path"] = str(produced)
                _atomic_write_json(manifest_path, manifest_data)
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(_process_chunk, idx): idx for idx in pending_indices
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"chunk[{idx}]: {exc}")

    if failures:
        raise RuntimeError(
            f"Upscaling failed for {len(failures)} chunk(s): " + "; ".join(failures)
        )

    return manifest_path
