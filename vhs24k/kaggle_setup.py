"""Kaggle environment bootstrapper for Real-ESRGAN, BasicSR, and GFPGAN (zero setup.py builds)."""

from __future__ import annotations

import pathlib
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

BASICSR_SDIST_URL = "https://files.pythonhosted.org/packages/source/b/basicsr/basicsr-1.4.2.tar.gz"

WEIGHT_URLS = {
    "weights/realesr-general-x4v3.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    ),
    "weights/realesr-general-wdn-x4v3.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
    ),
    "gfpgan/weights/GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    ),
}


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def extract_and_patch_basicsr(target_site_dir: pathlib.Path) -> pathlib.Path:
    """Download basicsr-1.4.2.tar.gz and install pure-Python package directly into site-packages without setup.py."""
    dest_pkg = target_site_dir / "basicsr"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        tar_path = tmp_path / "basicsr-1.4.2.tar.gz"
        urllib.request.urlretrieve(BASICSR_SDIST_URL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp_path)

        src_pkg = tmp_path / "basicsr-1.4.2" / "basicsr"
        if dest_pkg.exists():
            shutil.rmtree(dest_pkg)
        shutil.copytree(src_pkg, dest_pkg)

    version_py = dest_pkg / "version.py"
    version_py.write_text(
        "__version__ = '1.4.2'\n__gitsha__ = 'unknown'\nversion_info = (1, 4, 2)\n",
        encoding="utf-8",
    )

    deg_py = dest_pkg / "data" / "degradations.py"
    if deg_py.exists():
        deg_py.write_text(
            deg_py.read_text(encoding="utf-8").replace(
                "torchvision.transforms.functional_tensor",
                "torchvision.transforms.functional",
            ),
            encoding="utf-8",
        )
    return dest_pkg


def install_patched_basicsr() -> pathlib.Path:
    """Install patched basicsr into the active environment's site-packages."""
    site_dir = pathlib.Path(site.getsitepackages()[0])
    return extract_and_patch_basicsr(site_dir)


def setup_kaggle_environment(realesrgan_dir: str | pathlib.Path = "/kaggle/working/Real-ESRGAN") -> None:
    """Install all dependencies, clone Real-ESRGAN, install BasicSR directly, and download model weights."""
    realesrgan_path = pathlib.Path(realesrgan_dir)

    print("1/4 Installing runtime dependencies (wheels)...")
    _pip("addict", "lmdb", "future", "scipy", "yapf", "filterpy", "numba", "gdown", "ffmpeg-python")

    print("2/4 Installing patched basicsr directly into site-packages (bypassing broken setup.py)...")
    install_patched_basicsr()

    print("3/4 Installing facexlib, gfpgan, and realesrgan pre-built wheels (--no-deps)...")
    _pip("--no-deps", "facexlib", "gfpgan", "realesrgan")

    if not realesrgan_path.exists():
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/xinntao/Real-ESRGAN.git", str(realesrgan_path)],
            check=True,
        )

    print("4/4 Downloading Real-ESRGAN & GFPGAN weights...")
    for rel_dest, url in WEIGHT_URLS.items():
        dest = realesrgan_path / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)

    import importlib
    importlib.invalidate_caches()
    import basicsr  # noqa: F401
    import gfpgan  # noqa: F401
    import realesrgan  # noqa: F401
    from basicsr.data import degradations  # noqa: F401

    print("Setup complete — all imports (basicsr, gfpgan, realesrgan) verified!")
