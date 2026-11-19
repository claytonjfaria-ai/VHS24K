"""Chunks are encoded to final 4K in the worker; assembly just joins them."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vhs24k import postprocess, upscale
from vhs24k.config import PipelineConfig
from vhs24k.encoding import build_encoder_spec
from vhs24k.postprocess import assemble_final
from vhs24k.upscale import _plan_final_encode, build_enhancer_worker_cmd, upscale_manifest

FULLSCREEN = "scale=3840:2160:flags=lanczos,setsar=1"


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    return PipelineConfig(
        input_dir=tmp_path / "in", work_dir=tmp_path / "work", output_dir=tmp_path / "out", **kw
    )


def _manifest(tmp_path: Path, **extra) -> tuple[Path, dict]:
    src = tmp_path / "work" / "VTS_01"
    (src / "chunks").mkdir(parents=True)
    (src / "upscaled").mkdir(parents=True)
    chunk_in = src / "chunks" / "chunk_0000.mp4"
    chunk_in.write_bytes(b"x" * 64)
    audio = src / "audio.mka"
    audio.write_bytes(b"a" * 64)
    manifest = {
        "source_name": "VTS_01",
        "video_info": {"dar": "4:3"},
        "audio_path": str(audio),
        "chunks": [
            {
                "index": 0,
                "input_path": str(chunk_in),
                "output_path": str(src / "upscaled" / "chunk_0000.mp4"),
                "status": "pending",
            }
        ],
        **extra,
    }
    path = src / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


class TestPlanFinalEncode:
    def test_dry_run_plans_hevc_and_dar_aware_scale(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, dry_run=True, target_4k_mode="pillarbox_uhd")
        plan = _plan_final_encode({"video_info": {"dar": "4:3"}}, cfg)
        assert plan is not None and plan["codec"] == "libx265"
        assert plan["scale_filter"].startswith("scale=2880:2160") and "pad=3840:2160" in plan["scale_filter"]

    def test_pinned_codec_is_reused_so_resumed_sessions_stay_compatible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            upscale, "resolve_encoder", lambda *a, **k: pytest.fail("must not re-probe when pinned")
        )
        pinned = {"codec": "hevc_nvenc", "scale_filter": FULLSCREEN}
        assert _plan_final_encode({"final_encode": pinned}, _cfg(tmp_path)) == pinned

    def test_not_used_when_quality_review_is_off(self, tmp_path: Path) -> None:
        assert _plan_final_encode({}, _cfg(tmp_path, quality_review=False)) is None


class TestWorkerCommand:
    def test_codec_flags_added_only_when_requested(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, crf=19, gpu_post=False)
        plain = build_enhancer_worker_cmd(Path("i.mp4"), Path("o.mp4"), cfg, Path("R"))
        assert "--codec" not in plain and "--no-gpu-post" in plain

        final = build_enhancer_worker_cmd(
            Path("i.mp4"), Path("o.mp4"), cfg, Path("R"), codec="libx265", scale_filter=FULLSCREEN
        )
        assert final[final.index("--codec") + 1] == "libx265"
        assert final[final.index("--crf") + 1] == "19"
        assert final[final.index("--preset") + 1] == "auto"
        assert final[final.index("--scale-filter") + 1] == FULLSCREEN


class TestUpscaleManifestPinsEncode:
    def test_manifest_records_final_encode_and_worker_gets_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest_path, _ = _manifest(tmp_path)
        cfg = _cfg(tmp_path, realesrgan_dir=tmp_path / "Real-ESRGAN", num_gpus=1)
        (tmp_path / "Real-ESRGAN").mkdir()
        monkeypatch.setattr(upscale, "detect_gpus", lambda: [0])
        monkeypatch.setattr(upscale, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17))
        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"video")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(upscale.subprocess, "run", fake_run)
        upscale_manifest(manifest_path, cfg)

        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved["final_encode"]["codec"] == "libx265"
        assert saved["final_encode"]["scale_filter"] == FULLSCREEN
        assert saved["chunks"][0]["status"] == "done"
        # each finished chunk records how it was encoded (assembly relies on this)
        assert saved["chunks"][0]["final_encode"] == saved["final_encode"]
        assert seen and seen[0][seen[0].index("--codec") + 1] == "libx265"


class TestAssemble:
    def _run_assemble(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_tag: dict | None = None
    ) -> list[str]:
        manifest_path, manifest = _manifest(tmp_path)
        Path(manifest["chunks"][0]["output_path"]).write_bytes(b"v" * 64)
        manifest["chunks"][0]["status"] = "done"
        if chunk_tag is not None:
            manifest["chunks"][0]["final_encode"] = chunk_tag
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(
            postprocess, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17, "medium")
        )
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(postprocess.subprocess, "run", fake_run)
        assemble_final(manifest_path, tmp_path / "out" / "VTS_01_4k.mp4", _cfg(tmp_path))
        assert len(calls) == 1
        return calls[0]

    def test_final_chunks_are_stream_copied_not_reencoded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cmd = self._run_assemble(
            tmp_path, monkeypatch, chunk_tag={"codec": "libx265", "scale_filter": FULLSCREEN}
        )
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-vf" not in cmd and "-crf" not in cmd
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
        assert cmd[cmd.index("-c:a") + 1] == "aac"

    def test_chunks_without_final_encode_take_the_reencode_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cmd = self._run_assemble(tmp_path, monkeypatch)
        assert cmd[cmd.index("-vf") + 1] == FULLSCREEN
        assert cmd[cmd.index("-c:v") + 1] == "libx265"

    def test_mixed_chunks_are_never_stream_copied(self) -> None:
        """Old-session or differently encoded chunks would corrupt a stream-copy concat."""
        x265 = {"codec": "libx265", "scale_filter": FULLSCREEN}
        uniform = postprocess._uniform_final_encode
        assert uniform([{"final_encode": x265}, {"final_encode": dict(x265)}]) == x265
        assert uniform([{"final_encode": x265}, {}]) is None  # one legacy chunk
        assert uniform([{"final_encode": x265}, {"final_encode": {**x265, "codec": "libx264"}}]) is None
        assert uniform([{"final_encode": x265}, {"final_encode": {**x265, "scale_filter": "scale=1:1"}}]) is None
        assert uniform([]) is None

    def test_resumed_run_with_legacy_chunk_takes_reencode_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest_path, manifest = _manifest(tmp_path)
        base = manifest["chunks"][0]
        legacy = {**base, "status": "done"}
        final = {
            **base, "index": 1, "status": "done",
            "final_encode": {"codec": "libx265", "scale_filter": FULLSCREEN},
        }
        for c in (legacy, final):
            Path(c["output_path"]).write_bytes(b"v" * 64)
        manifest["chunks"] = [legacy, final]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(postprocess, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17))
        calls: list[list[str]] = []
        monkeypatch.setattr(
            postprocess.subprocess, "run",
            lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        assemble_final(manifest_path, tmp_path / "out" / "o.mp4", _cfg(tmp_path))
        assert "-vf" in calls[0] and calls[0][calls[0].index("-c:v") + 1] == "libx265"

    def test_reencode_falls_back_to_x264_when_x265_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest_path, manifest = _manifest(tmp_path)
        Path(manifest["chunks"][0]["output_path"]).write_bytes(b"v" * 64)
        manifest["chunks"][0]["status"] = "done"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(postprocess, "resolve_encoder", lambda *a, **k: build_encoder_spec("libx265", 17))
        codecs: list[str] = []

        def fake_run(cmd, **kwargs):
            codec = cmd[cmd.index("-c:v") + 1]
            codecs.append(codec)
            if codec == "libx265":
                raise subprocess.CalledProcessError(1, cmd, stderr="x265 missing")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(postprocess.subprocess, "run", fake_run)
        assemble_final(manifest_path, tmp_path / "out" / "o.mp4", _cfg(tmp_path))
        assert codecs == ["libx265", "libx264"]


# ---------------------------------------------------------------- end-to-end worker run
try:  # only the worker run below needs these; the tests above must run without them
    import cv2
    import torch  # noqa: F401

    HAVE_ML_LIBS = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAVE_ML_LIBS = False

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None or not HAVE_ML_LIBS,
    reason="needs ffmpeg, OpenCV and torch",
)


def _probe(path: Path, entries: str) -> list[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", f"stream={entries}", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out.split(",")


class _StubUpsampler:
    """Stands in for Real-ESRGAN: plain 4x resize."""

    def enhance(self, frame, outscale=4):
        return cv2.resize(frame, None, fx=outscale, fy=outscale, interpolation=cv2.INTER_CUBIC), None


@needs_ffmpeg
class TestWorkerEndToEnd:
    @pytest.fixture()
    def chunk(self, tmp_path: Path) -> Path:
        path = tmp_path / "chunk_0000.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc2=size=160x120:rate=25:duration=2", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
        return path

    def _run(self, tmp_path: Path, chunk: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]):
        from vhs24k import enhancer_worker
        from vhs24k.gpu_post import GpuPostProcessor

        tiles: list[int] = []

        def fake_init(args, tile):
            tiles.append(tile)
            return _StubUpsampler(), None

        monkeypatch.setattr(enhancer_worker, "_init_upsampler", fake_init)
        monkeypatch.setattr("vhs24k.gpu_post.make_post_processor", lambda use_gpu: GpuPostProcessor("cpu"))
        out = tmp_path / "up" / "chunk_0000.mp4"
        args = enhancer_worker.build_parser().parse_args(
            ["-i", str(chunk), "-o", str(out), "--report-dir", str(tmp_path / "up"),
             "--target-quality-score", "50", *extra]
        )
        return enhancer_worker.process_chunk(args), tiles

    def test_final_encode_has_every_frame_right_size_and_colour_tags(
        self, tmp_path: Path, chunk: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, tiles = self._run(
            tmp_path, chunk, monkeypatch,
            ["--codec", "libx264", "--crf", "20", "--scale-filter", "scale=320:240:flags=lanczos,setsar=1"],
        )
        width, height, frames = _probe(result, "width,height,nb_read_frames")
        assert (int(width), int(height)) == (320, 240)
        assert int(frames) == 50  # nothing dropped or duplicated by the pipeline
        assert _probe(result, "color_space")[0] == "bt709"
        assert tiles == [0], "SD frames must not be tiled"
        # calibration review ran once up-front and left its evidence behind
        report = json.loads((tmp_path / "up" / "chunk_0000_quality_report.json").read_text(encoding="utf-8"))
        assert {"before", "after", "recipe"} <= report.keys()
        assert (tmp_path / "up" / "chunk_0000_before_vs_after_4k.jpg").is_file()

    def test_pinned_codec_that_stopped_working_fails_loudly_instead_of_switching(
        self, tmp_path: Path, chunk: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vhs24k import enhancer_worker

        monkeypatch.setattr(enhancer_worker, "encoder_works", lambda codec, args=(): False)
        with pytest.raises(RuntimeError, match="refusing to switch codecs"):
            self._run(tmp_path, chunk, monkeypatch, ["--codec", "hevc_nvenc", "--scale-filter", FULLSCREEN])

    def test_legacy_mode_keeps_native_4x_size(
        self, tmp_path: Path, chunk: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = self._run(tmp_path, chunk, monkeypatch, [])
        width, height, frames = _probe(result, "width,height,nb_read_frames")
        # 160x120 minus the locked overscan crop (156x114), upscaled 4x
        assert (int(width), int(height)) == (624, 456)
        assert int(frames) == 50

    def test_without_quality_review_frames_pass_straight_through(
        self, tmp_path: Path, chunk: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = self._run(tmp_path, chunk, monkeypatch, ["--no-quality-review"])
        width, height, frames = _probe(result, "width,height,nb_read_frames")
        assert (int(width), int(height)) == (640, 480)
        assert int(frames) == 50
