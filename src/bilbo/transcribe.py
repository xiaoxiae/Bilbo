from __future__ import annotations

from pathlib import Path

import click

from .models import Segment, Word


def transcribe(
    audio_path: Path,
    lang: str,
    model_size: str = "large-v3-turbo",
    device: str = "auto",
) -> list[Segment]:
    from faster_whisper import WhisperModel

    click.echo(f"  Loading Whisper model ({model_size}) on {device}...")
    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    click.echo(f"  Transcribing {audio_path.name} ({lang})...")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    duration = info.duration
    segments = []
    for seg in segments_iter:
        words = []
        if seg.words:
            words = [Word(start=w.start, end=w.end, word=w.word.strip()) for w in seg.words]
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words))
        pct = min(100, seg.end / duration * 100) if duration > 0 else 0
        click.echo(f"\r  Transcribing: {pct:5.1f}% ({seg.end:.0f}/{duration:.0f}s)", nl=False)

    click.echo(f"\r  Transcribed {len(segments)} segments ({duration:.0f}s audio)        ")
    return segments
