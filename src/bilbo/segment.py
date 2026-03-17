from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pysbd

from .models import Segment, SegmentedText, Word

if TYPE_CHECKING:
    from .log import PipelineLog


def _words_to_sentences(
    words: list[Word], lang: str
) -> list[Segment]:
    if not words:
        return []

    # Record each word's char offset in the space-joined text
    word_offsets: list[int] = []
    offset = 0
    for w in words:
        word_offsets.append(offset)
        offset += len(w.word) + 1  # +1 for space separator

    full_text = " ".join(w.word for w in words)

    # Replace guillemets with ASCII quotes so pySBD treats them as sentence
    # boundaries (it recognizes "..." but not «...»). Since «, », and " are
    # each one Unicode code point, character offsets are preserved.
    seg_text = full_text.replace("«", '"').replace("»", '"')

    segmenter = pysbd.Segmenter(language=lang, clean=False, char_span=True)
    spans = segmenter.segment(seg_text)

    sentences: list[Segment] = []
    wi = 0  # forward word pointer

    for span in spans:
        text = full_text[span.start:span.end].strip()
        if not text:
            continue

        idx = span.start
        sent_end = span.end

        # Advance to first word overlapping this sentence
        while wi < len(words) and word_offsets[wi] + len(words[wi].word) <= idx:
            wi += 1

        if wi >= len(words):
            break

        first_wi = wi

        # Advance to last word overlapping this sentence
        last_wi = wi
        while last_wi + 1 < len(words) and word_offsets[last_wi + 1] < sent_end:
            last_wi += 1

        sentences.append(Segment(
            start=round(words[first_wi].start, 3),
            end=round(words[last_wi].end, 3),
            text=text,
            words=words[first_wi : last_wi + 1],
        ))

    return sentences


def segment_text(
    raw_segments: list[Segment],
    lang: str,
    log: PipelineLog | None = None,
) -> SegmentedText:
    if log:
        log.info(f"Segmenting {lang}...")

    # Per-segment: contain any pySBD quirks to individual segments
    sentences = []
    for seg in raw_segments:
        words = seg.words
        if not words:
            words = [Word(start=seg.start, end=seg.end, word=seg.text)]
        sentences.extend(_words_to_sentences(words, lang))

    if log:
        log.info(f"{lang}: {len(sentences)} sentences")

    return SegmentedText(sentences=sentences)


def _decode_to_wav(audio_path: Path) -> tuple[np.ndarray, int]:
    """Decode audio to 16 kHz mono float32 via ffmpeg, return (samples, sr)."""
    import soundfile as sf

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_f32le",
                tmp,
            ],
            capture_output=True,
            check=True,
        )
        data, sr = sf.read(tmp, dtype="float32")
        return data, sr
    finally:
        os.unlink(tmp)


_VAD_PADDING = 0.05  # 50 ms guard on each side of matched speech region


def refine_timestamps(
    segmented: SegmentedText,
    audio_path: Path,
    on_progress: Callable[[float, float | None], None] | None = None,
    _vad_result: list[dict[str, float]] | None = None,
) -> tuple[SegmentedText, dict]:
    """Refine each segment's start and end using Silero VAD (ownership-based).

    Each VAD speech region is assigned to the segment that contains more than
    50% of the region's duration.  Both .start and .end are then snapped to
    the extremes of the matched regions, with ±50 ms padding and clamped to
    the adjacent segment boundaries.

    Pass ``_vad_result`` directly to skip VAD inference (used in tests).

    Returns ``(refined_segmented_text, stats_dict)``.
    """
    sents = segmented.sentences
    if not sents:
        return segmented, {
            "adjusted": 0, "extended": 0, "contracted": 0,
            "total": 0, "avg_extend_ms": 0.0, "avg_contract_ms": 0.0,
        }

    # Get VAD speech regions
    if _vad_result is not None:
        speech_regions = _vad_result
    else:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        samples, sr = _decode_to_wav(audio_path)
        model = load_silero_vad()
        tensor = torch.from_numpy(samples)
        vad_cb = (lambda pct: on_progress(pct, None)) if on_progress else None
        speech_regions = get_speech_timestamps(
            tensor, model, sampling_rate=sr, return_seconds=True,
            progress_tracking_callback=vad_cb,
        )

    adjusted = 0
    deltas: list[float] = []

    for i, seg in enumerate(sents):
        prev_end = sents[i - 1].end if i > 0 else 0.0
        next_start = sents[i + 1].start if i < len(sents) - 1 else float("inf")

        # Ownership: keep VAD regions where >50% of their duration falls inside seg
        matched = []
        for region in speech_regions:
            r_start, r_end = region["start"], region["end"]
            dur = r_end - r_start
            if dur <= 0:
                continue
            overlap = max(0.0, min(r_end, seg.end) - max(r_start, seg.start))
            if overlap / dur > 0.5:
                matched.append(region)

        if not matched:
            continue

        new_start = max(min(r["start"] for r in matched) - _VAD_PADDING, prev_end)
        new_end = min(max(r["end"] for r in matched) + _VAD_PADDING, next_start)

        seg.start = new_start
        if seg.words:
            seg.words[0].start = new_start

        if new_end != seg.end:
            deltas.append((new_end - seg.end) * 1000)
            adjusted += 1
        seg.end = new_end
        if seg.words:
            seg.words[-1].end = new_end

    extended = sum(1 for d in deltas if d > 0)
    contracted = sum(1 for d in deltas if d < 0)
    stats = {
        "adjusted": adjusted,
        "extended": extended,
        "contracted": contracted,
        "total": len(sents),
        "avg_extend_ms": (
            sum(d for d in deltas if d > 0) / extended if extended else 0.0
        ),
        "avg_contract_ms": (
            sum(abs(d) for d in deltas if d < 0) / contracted if contracted else 0.0
        ),
    }
    return segmented, stats
