# VHS24K

VHS/DVD to 4K restoration and upscaling pipeline built with Python 3.10+, FFmpeg, and Real-ESRGAN (`realesr-general-x4v3` + GFPGAN face enhancement), designed for dual-GPU Kaggle Notebooks (`2x NVIDIA T4`).

## Architecture

VHS24K processes interlaced VHS/DVD rips (`.VOB` title sets or `.mp4` files) in four resumable stages coordinated by a per-source `manifest.json`:

1. **Source Discovery & Preprocessing (`vhs24k/preprocess.py`)**
   - `discover_sources()` groups DVD `.VOB` titles (`VTS_01_1.VOB`, `VTS_02_1..3.VOB`) while skipping menus (`VIDEO_TS.VOB`, `*_0.VOB`) and `<1 MB` stubs, or discovers `.mp4` files.
   - `prepare_chunks()` probes stream metadata (`ffprobe`), applies `bwdif` deinterlacing + mild `hqdn3d` denoising, splits video into `chunk_seconds` (default `60s`) segments (`libx264` CRF 10), extracts the original audio stream (`audio.mka`) without re-encoding, and writes `work/<source_name>/manifest.json`.
2. **Dual-GPU Upscaling (`vhs24k/upscale.py`)**
   - `detect_gpus()` enumerates available CUDA devices via `nvidia-smi -L` (or PyTorch).
   - `upscale_manifest()` dispatches pending chunks in parallel across `num_gpus` workers (binding each worker via `CUDA_VISIBLE_DEVICES`), running `inference_realesrgan_video.py` with `--tile 512`, `--dn`, and optional `--face_enhance` (GFPGAN).
   - Updates `manifest.json` atomically (`tempfile` + `os.fsync` + `os.replace`) after every completed chunk.
3. **4K Assembly & Audio Muxing (`vhs24k/postprocess.py`)**
   - `assemble_final()` concatenates upscaled chunks, applies `build_4k_scale_filter()` (`pillarbox_uhd` 2880x2160 padded to 3840x2160 for 4:3 sources, or `native_4x`), muxes the original audio (`aac` 192k), and encodes to `libx265` (with `libx264` fallback).
4. **Google Drive I/O & CLI (`vhs24k/drive_io.py`, `vhs24k/cli.py`, `vhs24k/config.py`)**
   - `PipelineConfig` validates all pipeline parameters.
   - `cli.py` exposes `inspect`, `download`, `prepare`, `upscale`, `assemble`, and `run` subcommands.

## Kaggle `GPU T4 x2` Setup

1. Open [notebooks/kaggle_vhs_to_4k.ipynb](notebooks/kaggle_vhs_to_4k.ipynb) in Kaggle.
2. In **Notebook Settings**:
   - Set **Accelerator** to **GPU T4 x2**.
   - Enable **Internet** (required to clone `xinntao/Real-ESRGAN`, install dependencies, and transfer files from Google Drive).
3. Verify both GPUs are visible (`!nvidia-smi` should list GPU 0 and GPU 1).
4. The pipeline defaults to `--num-gpus 2`, `--tile 512`, FP16 inference, and `realesr-general-x4v3` so each T4 processes independent 60-second chunks concurrently.

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
