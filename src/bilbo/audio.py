from __future__ import annotations

import subprocess
import tempfile
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
            "-f", "wav", "-acodec", "pcm_f32le",
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


def export_audio(data: np.ndarray, sr: int, path: Path, fmt: str = "m4b") -> None:
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), data, sr, subtype="FLOAT")

    if fmt == "wav":
        tmp.rename(path)
        return

    codec = "aac" if fmt == "m4b" else "libmp3lame"
    ext = f".{fmt}"
    out = path.with_suffix(ext)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp),
            "-c:a", codec, "-b:a", "64k",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    tmp.unlink()
