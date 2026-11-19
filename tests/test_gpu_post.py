"""The GPU post-processing must reproduce the numpy/OpenCV reference."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from vhs24k.gpu_post import CpuPostProcessor, GpuPostProcessor, gaussian_kernel_1d, make_post_processor
from vhs24k.quality_review import (
    EnhancementRecipe,
    apply_post_upscale_4k_camera_polish,
    temporal_smooth_4k_frame,
)


def _textured_frame(height: int = 96, width: int = 128, seed: int = 7) -> np.ndarray:
    """Frame with flat regions, hard edges, gradients and noise, so every branch is exercised."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    third = width // 3
    frame[:, :third] = (40, 90, 160)
    frame[:, third : 2 * third] = np.linspace(20, 235, third, dtype=np.uint8)[None, :, None]
    frame[:, 2 * third :] = (200, 210, 220)
    frame[height // 3 : 2 * height // 3, width // 2 : width // 2 + 6] = 250
    noise = rng.integers(-12, 13, size=frame.shape)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("ksize", [3, 5, 7, 11])
def test_gaussian_kernel_matches_opencv(ksize: int) -> None:
    ours = np.array(gaussian_kernel_1d(ksize), dtype=np.float64)
    theirs = cv2.getGaussianKernel(ksize, 0).ravel()
    assert ours.sum() == pytest.approx(1.0)
    assert np.allclose(ours, theirs, atol=1e-6)


def test_gaussian_kernel_rejects_even_sizes() -> None:
    with pytest.raises(ValueError):
        gaussian_kernel_1d(4)


@pytest.mark.parametrize("radius", [1, 2, 3, 5])
def test_blur_matches_cv2_including_borders(radius: int) -> None:
    proc = GpuPostProcessor("cpu")
    plane = np.random.default_rng(3).uniform(0, 255, size=(64, 80)).astype(np.float32)
    expected = cv2.GaussianBlur(plane, (2 * radius + 1, 2 * radius + 1), 0)
    actual = proc._blur(torch.from_numpy(plane), radius).numpy()
    assert np.abs(actual - expected).max() < 1e-3


@pytest.mark.parametrize(
    "recipe",
    [
        EnhancementRecipe(),
        EnhancementRecipe(micro_contrast=0.8, second_pass_sharpen=0.6),
    ],
)
def test_polish_matches_numpy_reference(recipe: EnhancementRecipe) -> None:
    frame = _textured_frame()
    expected = apply_post_upscale_4k_camera_polish(frame, recipe)
    actual = GpuPostProcessor("cpu").polish_to_numpy(frame, recipe)

    assert actual.shape == expected.shape and actual.dtype == np.uint8
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    # float32 rounding can flip a value by one level at an exact .5 boundary, never more
    assert diff.max() <= 1
    assert diff.mean() < 0.05


def test_temporal_matches_numpy_reference() -> None:
    proc = GpuPostProcessor("cpu")
    prev = _textured_frame(seed=1)
    curr = np.clip(_textured_frame(seed=1).astype(np.int16) + 6, 0, 255).astype(np.uint8)
    expected = temporal_smooth_4k_frame(curr, prev, strength=0.48)
    actual = proc.to_numpy(proc.temporal(proc.upload(curr), proc.upload(prev), 0.48))
    assert np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max() <= 1


def test_temporal_without_previous_frame_is_passthrough() -> None:
    proc = GpuPostProcessor("cpu")
    curr = proc.upload(_textured_frame())
    assert proc.temporal(curr, None, 0.48) is curr
    assert proc.temporal(curr, curr, 0.0) is curr


def test_make_post_processor_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert isinstance(make_post_processor(True), CpuPostProcessor)
    assert isinstance(make_post_processor(False), CpuPostProcessor)
