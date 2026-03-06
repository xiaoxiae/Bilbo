from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Segment, Word

if TYPE_CHECKING:
    from faster_whisper import BatchedInferencePipeline


def load_whisper_model(
    model_size: str = "large-v3-turbo",
    device: str = "auto",
) -> BatchedInferencePipeline:
    from faster_whisper import BatchedInferencePipeline as _BatchedPipeline
    from faster_whisper import WhisperModel as _WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    whisper = _WhisperModel(model_size, device=device, compute_type=compute_type)
    return _BatchedPipeline(whisper)


def transcribe(
    audio_path: Path,
    lang: str,
    model_size: str = "large-v3-turbo",
    device: str = "auto",
    model: BatchedInferencePipeline | None = None,
    batch_size: int | None = None,
    on_progress: Callable[[float, float | None], None] | None = None,
) -> list[Segment]:
    if model is None:
        model = load_whisper_model(model_size, device)

    if batch_size is None:
        batch_size = 16

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        batch_size=batch_size,
    )

    duration = info.duration
    segments = []
    for seg in segments_iter:
        words = []
        if seg.words:
            words = [Word(start=w.start, end=w.end, word=w.word.strip()) for w in seg.words]
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words))
        if on_progress:
            on_progress(seg.end, duration)

    return segments
