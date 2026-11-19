# VHS24K

VHS/DVD to 4K restoration and upscaling pipeline built with Python 3.10+, FFmpeg, Real-ESRGAN (`realesr-general-x4v3` WDN + `GFPGANv1.4` face enhancement), and an Automated AI Quality Reviewer (`vhs24k/quality_review.py`), designed for dual-GPU Kaggle Notebooks (`2x NVIDIA T4`).

## Architecture

VHS24K processes interlaced VHS/DVD rips (`.VOB` title sets or `.mp4` files) in four resumable stages coordinated by a per-source `manifest.json`:

1. **Source Discovery & Preprocessing (`vhs24k/preprocess.py`)**
   - `discover_sources()` groups DVD `.VOB` titles (`VTS_01_1.VOB`, `VTS_02_1..3.VOB`) while skipping menus (`VIDEO_TS.VOB`, `*_0.VOB`) and `<1 MB` stubs, or discovers `.mp4` files.
   - `prepare_chunks()` probes stream metadata (`ffprobe`), applies `bwdif` deinterlacing + temporal stabilization (`atadenoise` + `deflicker`) + `hqdn3d` denoising, splits video into `chunk_seconds` (default `60s`) segments (`libx264` CRF 10), extracts the original audio stream (`audio.mka`) without re-encoding, and writes `work/<source_name>/manifest.json`.
2. **Dual-GPU Upscaling & AI Quality Review (`vhs24k/upscale.py`, `vhs24k/enhancer_worker.py`, `vhs24k/quality_review.py`)**
   - `detect_gpus()` enumerates available CUDA devices via `nvidia-smi -L` (or PyTorch).
   - `upscale_manifest()` dispatches pending chunks in parallel across `num_gpus` workers (binding each worker via `CUDA_VISIBLE_DEVICES`), invoking `vhs24k.enhancer_worker` with `realesr-general-x4v3` WDN (`--denoise-strength 0.65`), `--face-weight 0.85` (`GFPGANv1.4`), and closed-loop AI quality review. SD frames are processed without tiling (no tile seams); tiling only applies to frames larger than `--no-tile-max` (1280 px).
   - Each worker encodes its chunk **straight to the final 4K format** (scale filter + encoder chosen once per source and pinned in `manifest.json` as `final_encode`, so resumed sessions stay compatible). The encode runs concurrently with GPU inference.
   - Updates `manifest.json` atomically (`tempfile` + `os.fsync` + `os.replace`) after every completed chunk.
3. **4K Assembly & Audio Muxing (`vhs24k/postprocess.py`)**
   - `assemble_final()` joins the already-final 4K chunks with a **stream copy** (no video re-encode, takes seconds) and muxes the original audio (`aac` 192k). Only chunks without a `final_encode` entry (e.g. `--no-quality-review`) take the legacy full re-encode path (`libx265` with `libx264` fallback).
   - `build_4k_scale_filter()` (default `--4k-mode fullscreen_uhd`: borderless 3840x2160 full-screen `lanczos` scaling with `setsar=1`; or `pillarbox_uhd` / `native_4x`) is applied inside the worker's encoder. RGB is converted to YUV with the BT.709 matrix the file is tagged with.
   - Encoder (`--codec`, default `auto`): `hevc_nvenc` (T4 hardware encoder) if FFmpeg can really use it, else `libx265`, else `libx264`. Each candidate is test-encoded with the exact arguments before use. `--crf` (default 17) sets quality.
4. **Google Drive I/O & CLI (`vhs24k/drive_io.py`, `vhs24k/cli.py`, `vhs24k/config.py`)**
   - `PipelineConfig` validates all pipeline parameters.
   - `cli.py` exposes `inspect`, `download`, `prepare`, `upscale`, `assemble`, and `run` subcommands.

## Automated AI Quality Reviewer & Closed-Loop 4K Camera Engine

`vhs24k/quality_review.py` and `vhs24k/enhancer_worker.py` implement a closed-loop perceptual restoration stage around each chunk:

1. **5-Dimension Perceptual Scoring (`0–100`)**: `analyze_frame()` evaluates each frame across:
   - **Sharpness** (multi-scale Laplacian & Tenengrad gradient energy)
   - **Face Clarity** (Haar face-patch detail or center-weighted texture)
   - **Dynamic Range** (luma p98–p2 spread, histogram entropy, and clipping penalty)
   - **Color Naturalness** (colorfulness + midtone gray-world neutrality)
   - **VHS Artifact-Free** (absence of interlace combing and YCrCb chroma noise)
2. **Pre-Upscale VHS Overscan Trimming, Bilateral Surface Smoothing & Color Fix**: `compute_safe_overscan_crop()` and `apply_overscan_crop()` conservatively trim black VHS overscan edges and bottom head-switching noise (`<= 4.5%` left/right, `<= 3.5%` top, `<= 5.5%` bottom, with even dimensions) before upscaling. `derive_enhancement_recipe()` and `apply_pre_upscale_color_and_lighting()` apply **Bilateral Surface Smoothing** (via `cv2.bilateralFilter` to clean VHS grain on flat surfaces while keeping edges sharp), remove YCrCb chroma bleed/noise, neutralize VHS color cast via highlight-protected Gray-World Auto White Balance, expand compressed dynamic range (adaptive LAB CLAHE), and apply skin-safe vibrance.
3. **`realesr-general-x4v3` WDN + `GFPGANv1.4` Restoration**: `enhancer_worker.py` upscales each frame 4× using `realesr-general-x4v3` with tunable WDN denoising (`--denoise-strength 0.65` by default) and blends restored faces with `GFPGANv1.4` (`--face-weight 0.85` by default).
4. **Structure-Gated 4K Edge Acutance (`edge_gate`), Motion-Gated Temporal Smoothing & Closed-Loop Refinement** (runs on the GPU via `vhs24k/gpu_post.py`, numerically matching the numpy reference in `quality_review.py`; `--no-gpu-post` falls back to the CPU path). The closed-loop review runs **once before the chunk's first frame** on a representative frame, so the refined recipe applies to every frame and the chunk has one consistent look. `apply_post_upscale_4k_camera_polish()` computes a structural gradient `edge_gate` (`grad` on smoothed luma) so flat sky/water surfaces (`edge_gate < 0.15`) are gently smoothed without amplifying noise while structural edges receive multi-scale 4K micro-contrast with halo suppression. `temporal_smooth_4k_frame()` applies motion-gated temporal anti-shimmer smoothing (`exp(-(diff/11)^2)`) so stationary micro-shimmer is stabilized while fast-moving edges stay sharp without ghosting. `closed_loop_review_and_refine()` re-scores the result against `--target-quality-score` (default `82.0`) and automatically triggers a second-pass refinement if below target.
5. **3-Stage VHS White-Line Dropout Eliminator & 3-Frame Temporal Median Dropout Inpainter**:
   - **Stage 1 (FFmpeg Preprocess):** `atadenoise=0a=0.06:0b=0.12:1a=0.06:1b=0.12:2a=0.06:2b=0.12` + `deflicker=size=5:mode=am` + `hqdn3d` stabilize tape grain and luminance flicker before upscaling.
   - **Stage 2 (3-Frame Temporal Median Dropout Inpainter):** `remove_vhs_white_line_dropouts()` detects horizontal white/bright VHS tape dropout streaks against adjacent frames (`prev_bgr`/`next_bgr`) and inpaints them cleanly before Real-ESRGAN runs.
   - **Stage 3 (Post-Upscale 4K Temporal Smoother):** `temporal_smooth_4k_frame()` suppresses frame-to-frame 4K hallucination shimmer on static regions while preserving motion.
6. **Diagnostics & Visual Proof**: For each chunk, `save_comparison_and_report()` writes `<chunk>_quality_report.json` (before/after scores, grade, fixed deficiencies, and recipe) and `<chunk>_before_vs_after_4k.jpg` (side-by-side comparison). Use `--no-quality-review` to bypass this stage if needed.

## Throughput Design

Per chunk, `vhs24k/frame_pipeline.py` runs three stages concurrently instead of one after another: **CPU prepare** (crop, dropout removal, colour fix) -> **GPU** (Real-ESRGAN, GFPGAN, 4K polish, temporal smoothing) -> **FFmpeg encode**. FFmpeg's stderr is drained continuously and run at `-loglevel error`; otherwise its progress output fills the pipe and silently freezes the worker. Any stage failing stops the others and surfaces the error.

Disk: Kaggle's `/kaggle/working` is limited to about 20 GB. Preprocessed chunks, the final chunks, and the joined MP4 all live there. Process one source at a time (`--source-name VTS_01`, then `VTS_02`) and download each finished MP4 before starting the next.

## Kaggle `GPU T4 x2` Setup

1. Open [notebooks/kaggle_vhs_to_4k.ipynb](notebooks/kaggle_vhs_to_4k.ipynb) in Kaggle.
2. In **Notebook Settings**:
   - Set **Accelerator** to **GPU T4 x2**.
   - Enable **Internet** (required to clone `xinntao/Real-ESRGAN`, install dependencies, and transfer files from Google Drive).
3. Verify both GPUs are visible (`!nvidia-smi` should list GPU 0 and GPU 1).
4. The pipeline defaults to `--num-gpus 2`, `--tile 512`, FP16 inference, and `realesr-general-x4v3` (`--denoise-strength 0.65`) so each T4 processes independent 60-second chunks concurrently.

## Google Drive `Wedding` Folder Integration

`vhs24k/drive_io.py` is pre-configured for the `Wedding` Google Drive folder (`DEFAULT_WEDDING_FOLDER_ID`) containing:
- **DVD VOBs (recommended, uncompressed MPEG-2):** `VTS_01_1.VOB`, `VTS_02_1.VOB`, `VTS_02_2.VOB`, `VTS_02_3.VOB`
- **MP4 copies:** `Wedding1.mp4` – `Wedding4.mp4`

### Transfer Methods
- **`rclone` (recommended for private Drive files):**
  Configure an `rclone` remote named `gdrive` (paste your `rclone.conf` token into Kaggle Secrets or `/root/.config/rclone/rclone.conf`), then download and upload:
  ```bash
  python -m vhs24k.cli download --input-dir /kaggle/working/input --source-mode vob --method rclone --rclone-remote gdrive:Wedding
  ```
  After upscaling, push finished 4K MP4s back to Drive with `upload_outputs(Path("/kaggle/working/output"), "gdrive:Wedding/4K_Output")`.
- **`gdown` (for link-shared files):**
  ```bash
  python -m vhs24k.cli download --input-dir /kaggle/working/input --source-mode vob --method gdown
  ```

## `--test-clip 15` Workflow

Before committing a full multi-hour upscale run, verify deinterlacing, 4:3 pillarboxing, denoise strength, and GFPGAN face enhancement on a 15-second clip:

```bash
python -m vhs24k.cli run \
  --input-dir /kaggle/working/input \
  --work-dir /kaggle/working/work \
  --output-dir /kaggle/working/output \
  --realesrgan-dir /kaggle/working/Real-ESRGAN \
  --test-clip 15
```

This trims preprocessing to the first 15 seconds (`-t 15`) and outputs `<source_name>_test_4k.mp4`. Once satisfied with the visual quality, remove `--test-clip 15` (and clear the test `work/` directory) to run the full tapes.

## 12-Hour Session Resume Behavior

Kaggle GPU sessions terminate after 12 hours. VHS24K is designed to resume cleanly across sessions:

- **Preprocessing skip:** If `work/<source_name>/manifest.json` already has `status` in `("prepared", "done")` and all chunk files + `audio.mka` exist on disk, `prepare_chunks()` returns immediately without re-running FFmpeg.
- **Chunk-level atomic checkpointing:** `upscale_manifest()` writes `manifest.json` atomically after each chunk finishes (`"status": "done"`). On restart, only chunks whose status is not `"done"` (i.e. `"pending"` or `"failed"`) are re-queued.
- **Resuming:** Persist `/kaggle/working/work` (e.g., via Kaggle Datasets or `rclone` sync) and re-run the exact same `run` or `upscale` command; completed chunks are skipped automatically.

## Running Tests

```bash
pytest -v
```
