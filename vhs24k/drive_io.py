"""Google Drive I/O helpers for VHS24K (rclone & gdown)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WEDDING_FOLDER_ID = "1y3TRQzoGhupnogwJAqPMilksRTKcz5S6"

WEDDING_FILES: dict[str, str] = {
    "VTS_01_1.VOB": "1oHSKJQxpYpyzOYQjtNfcSIkvL9MwC2gK",
    "VTS_02_1.VOB": "1ffas4R6CvVGbkkM5egXbxQn3CkZ_gyzK",
    "VTS_02_2.VOB": "1zKD4chAI5lvxhjvKbErb73dGfcC3IqOp",
    "VTS_02_3.VOB": "1TVE2E0yqvBvRytj-z3DHHrTcfQRSh055",
    "Wedding1.mp4": "1zq-W1Timup9bP8PilE2r6_RxgPE9uiOf",
    "Wedding2.mp4": "1LK0L8zQUXsL1gMxl9LNPkjOAFYb3p9Gh",
    "Wedding3.mp4": "1GdGwKPfoveYA6lkno84OpiyupDaMLpre",
    "Wedding4.mp4": "1JqJnpDaxXGOVTj-niEgjxtcaTJNGgJJH",
}

_SOURCE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "vob": (".vob",),
    "mp4": (".mp4",),
    "all": (".vob", ".mp4"),
}


def _select_wedding_files(source_type: str) -> dict[str, str]:
    """Filter ``WEDDING_FILES`` by ``source_type`` (``vob``, ``mp4``, or ``all``)."""
    key = source_type.lower()
    if key not in _SOURCE_EXTENSIONS:
        raise ValueError(
            f"Unsupported source_type '{source_type}'. Expected one of: vob, mp4, all."
        )
    exts = _SOURCE_EXTENSIONS[key]
    return {
        name: file_id
        for name, file_id in WEDDING_FILES.items()
        if name.lower().endswith(exts)
    }


def _require_rclone() -> str:
    """Return path to ``rclone`` executable or raise ``FileNotFoundError``."""
    rclone_bin = shutil.which("rclone")
    if not rclone_bin:
        raise FileNotFoundError(
            "rclone is not installed or not on PATH. Install rclone and configure a "
            "Google Drive remote (e.g. 'gdrive') before using rclone transfer."
        )
    return rclone_bin


def download_wedding_files(
    dest_dir: Path,
    source_type: str = "vob",
    method: str = "rclone",
    rclone_remote: str = "gdrive:Wedding",
) -> list[Path]:
    """Download wedding source files from Google Drive into ``dest_dir``.

    Args:
        dest_dir: Local destination directory.
        source_type: ``"vob"``, ``"mp4"``, or ``"all"``.
        method: ``"rclone"`` (recommended for private Drive files) or ``"gdown"``
            (for link-shared files).
        rclone_remote: rclone remote path (default ``"gdrive:Wedding"``).

    Returns:
        List of downloaded ``Path`` objects present in ``dest_dir``.
    """
    selected = _select_wedding_files(source_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    transfer_method = method.lower()

    if transfer_method == "rclone":
        rclone_bin = _require_rclone()
        cmd = [
            rclone_bin,
            "copy",
            rclone_remote,
            str(dest_dir),
            "--progress",
            "--transfers",
            "4",
        ]
        for filename in selected:
            cmd.extend(["--include", filename])

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"rclone copy failed with exit code {proc.returncode}: {' '.join(cmd)}"
            )
        return [dest_dir / name for name in selected if (dest_dir / name).exists()]

    if transfer_method == "gdown":
        try:
            import gdown  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "gdown is not installed. Install it with 'pip install gdown'."
            ) from exc

        downloaded: list[Path] = []
        for filename, file_id in selected.items():
            out_path = dest_dir / filename
            if out_path.exists() and out_path.stat().st_size > 0:
                logger.info("Skipping existing file: %s", out_path)
                downloaded.append(out_path)
                continue

            url = f"https://drive.google.com/uc?id={file_id}"
            result = gdown.download(url=url, output=str(out_path), quiet=False)
            if not result or not out_path.exists():
                raise RuntimeError(
                    f"gdown failed to download {filename} (id={file_id}). "
                    "Ensure the file is link-shared or use method='rclone'."
                )
            downloaded.append(out_path)
        return downloaded

    raise ValueError(
        f"Unsupported download method '{method}'. Expected 'rclone' or 'gdown'."
    )


def upload_outputs(
    src_dir: Path,
    rclone_remote: str = "gdrive:Wedding/4K_Output",
) -> None:
    """Upload rendered 4K output files from ``src_dir`` to Google Drive via ``rclone``."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {src_dir}")

    rclone_bin = _require_rclone()
    cmd = [
        rclone_bin,
        "copy",
        str(src_dir),
        rclone_remote,
        "--progress",
        "--transfers",
        "4",
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rclone upload failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
