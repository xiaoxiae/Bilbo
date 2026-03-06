from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import pyloudnorm


TARGET_LUFS = -16.0


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        return data, sr
    except sf.LibsndfileError:
        pass
    # Fallback: decode via ffmpeg to WAV in memory
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-f", "wav", "-acodec", "pcm_s16le", "-"],
        capture_output=True,
        check=True,
    )
    import io
    data, sr = sf.read(io.BytesIO(result.stdout), dtype="float64", always_2d=True)
    return data, sr


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    from scipy.signal import resample as scipy_resample
    num_samples = int(len(data) * target_sr / orig_sr)
    return scipy_resample(data, num_samples)


def slice_audio(
    data: np.ndarray,
    sr: int,
    start: float,
    end: float,
    padding_ms: float = 75,
) -> np.ndarray:
    pad_s = padding_ms / 1000.0
    s = max(0, int((start - pad_s) * sr))
    e = min(len(data), int((end + pad_s) * sr))
    return data[s:e]


def normalize_lufs(data: np.ndarray, sr: int, target: float = TARGET_LUFS) -> np.ndarray:
    if len(data) == 0:
        return data
    meter = pyloudnorm.Meter(sr)
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError:
        return data
    if np.isinf(loudness):
        return data
    return pyloudnorm.normalize.loudness(data, loudness, target)


def crossfade(a: np.ndarray, b: np.ndarray, ms: int = 30) -> np.ndarray:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    samples = min(ms * 48, len(a), len(b))  # approximate, assumes <=48kHz
    if samples < 2:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, samples).reshape(-1, 1)
    fade_in = np.linspace(0.0, 1.0, samples).reshape(-1, 1)
    overlap = a[-samples:] * fade_out + b[:samples] * fade_in
    return np.concatenate([a[:-samples], overlap, b[samples:]])


def generate_silence(sr: int, ms: int, channels: int = 2) -> np.ndarray:
    samples = int(sr * ms / 1000)
    return np.zeros((samples, channels))


def export_audio(data: np.ndarray, sr: int, path: Path, fmt: str = "m4b") -> None:
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), data, sr)

    if fmt == "wav":
        tmp.rename(path)
        return

    import subprocess
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
