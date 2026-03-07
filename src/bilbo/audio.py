from __future__ import annotations

import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_LUFS = -16.0


def _probe_duration(path: Path) -> float | None:
    """Get audio duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def preprocess_audio(
    input_path: Path,
    target_sr: int = 24000,
    target_lufs: float = TARGET_LUFS,
    on_progress: Callable[[float, float | None], None] | None = None,
) -> Path:
    """Resample + LUFS-normalize via ffmpeg in a single streaming pass.

    Returns path to a temporary WAV file (float32, random-access compatible).
    Caller is responsible for cleanup.

    If *on_progress* is provided, it is called with (current_secs, total_secs | None)
    on each ffmpeg progress update.
    """
    import os

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)

    duration = _probe_duration(input_path)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(target_sr),
            "-af", f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5",
            "-f", "wav", "-acodec", "pcm_f32le", "-rf64", "auto",
            "-progress", "pipe:1", "-nostats",
            tmp_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                us = int(line.split("=", 1)[1])
            except ValueError:
                continue
            secs = us / 1_000_000
            if on_progress is not None:
                on_progress(secs, duration)

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise subprocess.CalledProcessError(proc.returncode, "ffmpeg", stderr=stderr)

    return Path(tmp_path)


def slice_audio(
    path: Path,
    sr: int,
    start: float,
    end: float,
    padding_ms: float = 75,
) -> np.ndarray:
    """Read only the needed slice from a WAV file (random-access, no full load)."""
    pad_s = padding_ms / 1000.0
    info = sf.info(str(path))
    total_frames = info.frames

    frame_s = max(0, int((start - pad_s) * sr))
    frame_e = min(total_frames, int((end + pad_s) * sr))
    if frame_e <= frame_s:
        return np.zeros((0, info.channels), dtype=np.float32)

    data, _ = sf.read(str(path), start=frame_s, stop=frame_e, dtype="float32", always_2d=True)
    return data


def crossfade(a: np.ndarray, b: np.ndarray, ms: int = 30) -> np.ndarray:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    samples = min(ms * 48, len(a), len(b))  # approximate, assumes <=48kHz
    if samples < 2:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, samples, dtype=np.float32).reshape(-1, 1)
    fade_in = np.linspace(0.0, 1.0, samples, dtype=np.float32).reshape(-1, 1)
    overlap = a[-samples:] * fade_out + b[:samples] * fade_in
    return np.concatenate([a[:-samples], overlap, b[samples:]])


def generate_silence(sr: int, ms: int, channels: int = 2) -> np.ndarray:
    samples = int(sr * ms / 1000)
    return np.zeros((samples, channels), dtype=np.float32)


def apply_fade(chunk: np.ndarray, sr: int, fade_ms: int = 5) -> np.ndarray:
    """Apply short fade-in and fade-out to a chunk to avoid clicks."""
    if len(chunk) == 0:
        return chunk
    fade_samples = min(int(sr * fade_ms / 1000), len(chunk) // 2)
    if fade_samples < 2:
        return chunk
    chunk = chunk.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32).reshape(-1, 1)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32).reshape(-1, 1)
    chunk[:fade_samples] *= fade_in
    chunk[-fade_samples:] *= fade_out
    return chunk


class AudioExporter:
    """Context manager that streams PCM chunks to ffmpeg for encoding.

    Usage::

        with AudioExporter(sr, channels, path, fmt) as exp:
            exp.write(chunk1)
            exp.write(chunk2)
        print(exp.duration)
    """

    def __init__(
        self,
        sr: int,
        channels: int,
        path: Path,
        fmt: str = "m4b",
        on_progress: Callable[[float, float | None], None] | None = None,
    ) -> None:
        self.sr = sr
        self.channels = channels
        self.path = path
        self.fmt = fmt
        self.on_progress = on_progress
        self.total_samples = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._progress_thread: threading.Thread | None = None

    @property
    def duration(self) -> float:
        return self.total_samples / self.sr

    def __enter__(self) -> AudioExporter:
        codec = "aac" if self.fmt == "m4b" else "libmp3lame"
        out = self.path.with_suffix(f".{self.fmt}")
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "f32le", "-ar", str(self.sr), "-ac", str(self.channels),
                "-i", "pipe:0",
                "-c:a", codec, "-b:a", "64k",
                "-progress", "pipe:1", "-nostats",
                str(out),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        self._progress_thread = threading.Thread(target=self._read_progress, daemon=True)
        self._progress_thread.start()
        return self

    def write(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        self.total_samples += len(chunk)
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(chunk.tobytes())

    def _read_progress(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("out_time_us=") and self.on_progress is not None:
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                self.on_progress(us / 1_000_000, self.total_samples / self.sr)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        assert self._proc is not None
        if self._proc.stdin:
            self._proc.stdin.close()
        if self._progress_thread:
            self._progress_thread.join()
        self._proc.wait()
        if exc_type is None and self._proc.returncode != 0:
            stderr = self._proc.stderr.read() if self._proc.stderr else b""
            raise subprocess.CalledProcessError(
                self._proc.returncode, "ffmpeg", stderr=stderr
            )
