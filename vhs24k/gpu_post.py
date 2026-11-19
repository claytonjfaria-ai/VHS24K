"""GPU (PyTorch) implementations of the per-frame 4K post-processing stages.

``apply_post_upscale_4k_camera_polish`` and ``temporal_smooth_4k_frame`` in
``quality_review.py`` run on multi-megapixel frames and dominated the CPU time
of every chunk. The functions here reproduce the same math on the GPU (OpenCV
Gaussian kernels and reflect-101 borders included), so output matches the numpy
reference to within rounding while costing milliseconds instead of seconds.

``CpuPostProcessor`` exposes the identical interface on top of the numpy
reference so the worker can fall back when no CUDA device is present.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vhs24k.quality_review import (
    EnhancementRecipe,
    apply_post_upscale_4k_camera_polish,
    temporal_smooth_4k_frame,
)

# OpenCV uses these exact fixed kernels for small odd sizes when sigma == 0.
_SMALL_GAUSSIAN_KERNELS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    3: (0.25, 0.5, 0.25),
    5: (0.0625, 0.25, 0.375, 0.25, 0.0625),
    7: (0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125),
}


def gaussian_kernel_1d(ksize: int) -> list[float]:
    """1-D Gaussian taps identical to ``cv2.GaussianBlur(arr, (k, k), 0)``."""
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")
    if ksize in _SMALL_GAUSSIAN_KERNELS:
        return list(_SMALL_GAUSSIAN_KERNELS[ksize])
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1.0) + 0.8
    half = (ksize - 1) / 2.0
    weights = [math.exp(-((i - half) ** 2) / (2.0 * sigma * sigma)) for i in range(ksize)]
    total = sum(weights)
    return [w / total for w in weights]


class GpuPostProcessor:
    """Runs polish + temporal smoothing on a torch device (CUDA or CPU)."""

    def __init__(self, device: Any = None) -> None:
        import torch  # local import: the core package must not require torch

        self._torch = torch
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._kernels: dict[int, Any] = {}

    # ----- helpers -------------------------------------------------------
    def _kernel(self, ksize: int) -> Any:
        if ksize not in self._kernels:
            taps = self._torch.tensor(gaussian_kernel_1d(ksize), dtype=self._torch.float32, device=self.device)
            self._kernels[ksize] = taps
        return self._kernels[ksize]

    def _blur(self, plane: Any, radius: int) -> Any:
        """Gaussian blur of an (H, W) float plane, matching cv2 (reflect-101 border)."""
        if radius <= 0:
            return plane.clone()
        torch = self._torch
        f = torch.nn.functional
        ksize = 2 * radius + 1
        taps = self._kernel(ksize)
        x = plane[None, None]
        x = f.pad(x, (radius, radius, 0, 0), mode="reflect")
        x = f.conv2d(x, taps.view(1, 1, 1, ksize))
        x = f.pad(x, (0, 0, radius, radius), mode="reflect")
        x = f.conv2d(x, taps.view(1, 1, ksize, 1))
        return x[0, 0]

    def _abs_central_diff(self, plane: Any, axis: int) -> Any:
        """0.5*|p[i+1]-p[i-1]| along ``axis`` with edge replication (matches np.pad 'edge')."""
        f = self._torch.nn.functional
        if axis == 1:
            d = (plane[:, 2:] - plane[:, :-2]).abs() * 0.5
            return f.pad(d[None, None], (1, 1, 0, 0), mode="replicate")[0, 0]
        d = (plane[2:, :] - plane[:-2, :]).abs() * 0.5
        return f.pad(d[None, None], (0, 0, 1, 1), mode="replicate")[0, 0]

    # ----- public API ----------------------------------------------------
    def upload(self, frame_bgr: np.ndarray) -> Any:
        """Copy an (H, W, 3) uint8 BGR numpy frame to the device."""
        return self._torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(self.device)

    def to_numpy(self, frame: Any) -> np.ndarray:
        """Copy a device uint8 frame back to a contiguous numpy array."""
        return np.ascontiguousarray(frame.cpu().numpy())

    def polish(self, frame_bgr: np.ndarray, recipe: EnhancementRecipe) -> Any:
        """GPU equivalent of ``apply_post_upscale_4k_camera_polish``; returns a device uint8 tensor."""
        torch = self._torch
        with torch.inference_mode():
            f = self.upload(frame_bgr).permute(2, 0, 1).to(torch.float32)
            b, g, r = f[0], f[1], f[2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = (r - y) * 0.713 + 128.0
            cb = (b - y) * 0.564 + 128.0

            y_base = self._blur(y, 3)
            grad = torch.hypot(self._abs_central_diff(y_base, 1), self._abs_central_diff(y_base, 0))
            edge_gate = ((grad - 3.5) / 12.0).clamp(0.0, 1.0)

            y_flat = self._blur(y, 1)
            y = torch.where(edge_gate < 0.15, 0.8 * y + 0.2 * y_flat, y)

            blur_fine = self._blur(y, 2)
            blur_med = self._blur(y, 5)
            fine_detail = y - blur_fine
            med_detail = blur_fine - blur_med

            g_fine = 1.35 * recipe.micro_contrast + 1.40 * recipe.second_pass_sharpen
            g_med = 0.85 * recipe.micro_contrast + 0.90 * recipe.second_pass_sharpen
            delta = g_fine * fine_detail + g_med * med_detail
            midtone = (torch.minimum(y, 255.0 - y) / 24.0).clamp(0.25, 1.0)
            delta = delta.clamp(-18.0, 18.0) * edge_gate * midtone
            y_out = (y + delta).clamp(0.0, 255.0)

            cr_c = cr - 128.0
            cb_c = cb - 128.0
            out_r = y_out + 1.403 * cr_c
            out_g = y_out - 0.714 * cr_c - 0.344 * cb_c
            out_b = y_out + 1.773 * cb_c
            out = torch.stack([out_b, out_g, out_r], dim=-1)
            return torch.round(out).clamp(0, 255).to(torch.uint8)

    def temporal(self, curr: Any, prev: Any | None, strength: float) -> Any:
        """GPU equivalent of ``temporal_smooth_4k_frame`` on device tensors."""
        if prev is None or prev.shape != curr.shape or strength <= 0.0:
            return curr
        torch = self._torch
        with torch.inference_mode():
            curr_f = curr.to(torch.float32)
            prev_f = prev.to(torch.float32)
            diff = (curr_f - prev_f).abs().mean(dim=2, keepdim=True)
            alpha = float(min(max(strength, 0.0), 0.6)) * torch.exp(-((diff / 11.0) ** 2))
            blended = curr_f * (1.0 - alpha) + prev_f * alpha
            return torch.round(blended).clamp(0, 255).to(torch.uint8)

    def polish_to_numpy(self, frame_bgr: np.ndarray, recipe: EnhancementRecipe) -> np.ndarray:
        """Convenience wrapper: numpy in, numpy out (used by the quality-review loop)."""
        return self.to_numpy(self.polish(frame_bgr, recipe))


class CpuPostProcessor:
    """Numpy/OpenCV fallback with the same interface as ``GpuPostProcessor``."""

    device = "cpu"

    def upload(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr

    def to_numpy(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def polish(self, frame_bgr: np.ndarray, recipe: EnhancementRecipe) -> np.ndarray:
        return apply_post_upscale_4k_camera_polish(frame_bgr, recipe)

    def temporal(self, curr: np.ndarray, prev: np.ndarray | None, strength: float) -> np.ndarray:
        return temporal_smooth_4k_frame(curr, prev, strength=strength)

    def polish_to_numpy(self, frame_bgr: np.ndarray, recipe: EnhancementRecipe) -> np.ndarray:
        return self.polish(frame_bgr, recipe)


def make_post_processor(use_gpu: bool) -> GpuPostProcessor | CpuPostProcessor:
    """Return the GPU processor when requested and CUDA is available, else the CPU fallback."""
    if use_gpu:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                return GpuPostProcessor("cuda")
        except ImportError:
            pass
    return CpuPostProcessor()
