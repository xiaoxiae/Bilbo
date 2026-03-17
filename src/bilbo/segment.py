from __future__ import annotations

import os
import subprocess
import tempfile
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


def _rms(samples: np.ndarray) -> float:
    """Root mean square of a sample array."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def refine_timestamps(
    segmented: SegmentedText,
    audio_path: Path,
    win_ms: int = 30,
    threshold: float = 0.01,
    max_extend_ms: int = 300,
    max_contract_ms: int = 300,
) -> tuple[SegmentedText, dict]:
    """Refine segment end timestamps by scanning for speech/silence boundaries.

    For each segment end, scans both backward (into the segment) and forward
    (past the boundary) in ``win_ms`` windows looking for the first silent
    window (RMS < threshold). Picks whichever boundary is closer to the
    original end and adjusts accordingly.

    Returns ``(refined_segmented_text, stats_dict)``.
    """
    samples, sr = _decode_to_wav(audio_path)
    win_samples = int(sr * win_ms / 1000)

    adjusted = 0
    extended = 0
    contracted = 0
    extend_deltas: list[float] = []
    contract_deltas: list[float] = []

    for seg in segmented.sentences:
        end_sample = int(seg.end * sr)
        start_sample = int(seg.start * sr)

        # --- backward scan: find where speech ends ---
        max_contract_samples = int(sr * max_contract_ms / 1000)
        backward_limit = max(start_sample, end_sample - max_contract_samples)
        backward_pos: int | None = None
        pos = end_sample
        while pos - win_samples >= backward_limit:
            pos -= win_samples
            window = samples[pos : pos + win_samples]
            if _rms(window) >= threshold:
                # Found speech — boundary is just past this window
                backward_pos = pos + win_samples
                break

        # --- forward scan (past boundary) ---
        max_extend_samples = int(sr * max_extend_ms / 1000)
        forward_limit = min(len(samples), end_sample + max_extend_samples)
        forward_pos: int | None = None
        pos = end_sample
        while pos + win_samples <= forward_limit:
            window = samples[pos : pos + win_samples]
            if _rms(window) < threshold:
                # Found silence — boundary is at start of this window
                forward_pos = pos
                break
            pos += win_samples

        # Discard candidates that don't actually move the boundary
        if backward_pos is not None and backward_pos == end_sample:
            backward_pos = None
        if forward_pos is not None and forward_pos == end_sample:
            forward_pos = None

        # Pick the closer boundary
        best_pos: int | None = None
        if backward_pos is not None and forward_pos is not None:
            back_dist = abs(end_sample - backward_pos)
            fwd_dist = abs(forward_pos - end_sample)
            best_pos = backward_pos if back_dist <= fwd_dist else forward_pos
        elif backward_pos is not None:
            best_pos = backward_pos
        elif forward_pos is not None:
            best_pos = forward_pos

        if best_pos is not None:
            new_end = round(best_pos / sr, 3)
            delta_ms = (new_end - seg.end) * 1000
            seg.end = new_end
            if seg.words:
                seg.words[-1].end = new_end
            adjusted += 1
            if delta_ms > 0:
                extended += 1
                extend_deltas.append(delta_ms)
            else:
                contracted += 1
                contract_deltas.append(abs(delta_ms))

    total = len(segmented.sentences)
    stats = {
        "adjusted": adjusted,
        "extended": extended,
        "contracted": contracted,
        "total": total,
        "avg_extend_ms": (
            sum(extend_deltas) / len(extend_deltas) if extend_deltas else 0.0
        ),
        "avg_contract_ms": (
            sum(contract_deltas) / len(contract_deltas) if contract_deltas else 0.0
        ),
    }
    return segmented, stats
