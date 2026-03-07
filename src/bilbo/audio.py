from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ChapterMarker

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


def generate_tone(
    sr: int,
    freq: float,
    duration_ms: int,
    channels: int,
    amplitude: float = 0.3,
    fade_ms: int = 30,
) -> np.ndarray:
    """Generate a sine wave tone with smooth fade-in/fade-out."""
    n_samples = int(sr * duration_ms / 1000)
    t = np.arange(n_samples, dtype=np.float32) / sr
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade_samples = min(int(sr * fade_ms / 1000), n_samples // 2)
    if fade_samples >= 2:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
    # Expand to (samples, channels)
    return np.stack([tone] * channels, axis=1)


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
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.sr = sr
        self.channels = channels
        self.path = path
        self.fmt = fmt
        self.on_progress = on_progress
        self.metadata = metadata
        self.total_samples = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._progress_thread: threading.Thread | None = None

    @property
    def duration(self) -> float:
        return self.total_samples / self.sr

    def __enter__(self) -> AudioExporter:
        codec = "aac" if self.fmt == "m4b" else "libmp3lame"
        out = self.path.with_suffix(f".{self.fmt}")

        meta_flags: list[str] = []
        if self.metadata:
            for key, value in self.metadata.items():
                meta_flags.extend(["-metadata", f"{key}={value}"])

        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "f32le", "-ar", str(self.sr), "-ac", str(self.channels),
                "-i", "pipe:0",
                "-c:a", codec, "-b:a", "64k",
                *meta_flags,
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


def post_process_metadata(
    audio_path: Path,
    cover_path: Path | None = None,
    chapters: list[ChapterMarker] | None = None,
) -> None:
    """Embed cover art and/or chapter markers into an existing audio file.

    For m4b: uses ffmetadata + ffmpeg remux.
    For mp3: uses ffmpeg remux for cover, mutagen for chapters.
    Atomic writes throughout (.tmp -> rename).
    """
    ext = audio_path.suffix.lower()
    if ext == ".m4b":
        _post_process_m4b(audio_path, cover_path, chapters)
    elif ext == ".mp3":
        _post_process_mp3(audio_path, cover_path, chapters)


def _post_process_m4b(
    audio_path: Path,
    cover_path: Path | None,
    chapters: list[ChapterMarker] | None,
) -> None:
    """Remux m4b with cover art and/or chapter metadata via ffmpeg."""
    if not cover_path and not chapters:
        return

    meta_file = None
    try:
        inputs = ["-i", str(audio_path)]
        maps = ["-map", "0:a"]
        codec_flags = ["-c:a", "copy"]
        extra: list[str] = []

        # Write ffmetadata file with chapters
        if chapters:
            fd, meta_path = tempfile.mkstemp(suffix=".txt")
            meta_file = Path(meta_path)
            lines = [";FFMETADATA1"]
            for ch in chapters:
                lines.append("")
                lines.append("[CHAPTER]")
                lines.append("TIMEBASE=1/1000")
                lines.append(f"START={ch.start_ms}")
                lines.append(f"END={ch.end_ms}")
                lines.append(f"title={ch.title}")
            os.write(fd, "\n".join(lines).encode("utf-8"))
            os.close(fd)
            inputs.extend(["-i", meta_path])
            extra.extend(["-map_metadata", str(len(inputs) // 2 - 1)])

        # Cover art
        if cover_path:
            inputs.extend(["-i", str(cover_path)])
            cover_idx = len(inputs) // 2 - 1
            maps.extend(["-map", f"{cover_idx}:v"])
            codec_flags.extend(["-c:v", "copy", f"-disposition:v:0", "attached_pic"])

        tmp_out = audio_path.with_suffix(".tmp.m4b")
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            *maps,
            *codec_flags,
            *extra,
            str(tmp_out),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        tmp_out.rename(audio_path)
    finally:
        if meta_file:
            meta_file.unlink(missing_ok=True)


def _post_process_mp3(
    audio_path: Path,
    cover_path: Path | None,
    chapters: list[ChapterMarker] | None,
) -> None:
    """Embed cover and chapters into mp3."""
    # Cover via ffmpeg remux
    if cover_path:
        tmp_out = audio_path.with_suffix(".tmp.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path), "-i", str(cover_path),
                "-map", "0:a", "-map", "1",
                "-c:a", "copy",
                "-id3v2_version", "3",
                str(tmp_out),
            ],
            capture_output=True,
            check=True,
        )
        tmp_out.rename(audio_path)

    # Chapters via mutagen ID3
    if chapters:
        from mutagen.id3 import ID3, CTOC, CHAP, TIT2, CTOCFlags

        tag = ID3(str(audio_path))

        # Remove existing chapter frames
        tag.delall("CHAP")
        tag.delall("CTOC")

        chap_ids = []
        for i, ch in enumerate(chapters):
            chap_id = f"chp{i}"
            chap_ids.append(chap_id)
            tag.add(CHAP(
                element_id=chap_id,
                start_time=ch.start_ms,
                end_time=ch.end_ms,
                start_offset=0xFFFFFFFF,
                end_offset=0xFFFFFFFF,
                sub_frames=[TIT2(encoding=3, text=[ch.title])],
            ))

        tag.add(CTOC(
            element_id="toc",
            flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
            child_element_ids=chap_ids,
            sub_frames=[TIT2(encoding=3, text=["Table of Contents"])],
        ))

        tag.save()
