from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pysbd
import soundfile as sf

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

    # Flatten all words from raw segments
    all_words = []
    for seg in raw_segments:
        all_words.extend(seg.words)

    if not all_words:
        # Fallback: if no word-level timestamps, create one word per segment
        if log:
            log.warn("no word-level timestamps, using segment-level fallback")
        all_words = [Word(start=seg.start, end=seg.end, word=seg.text) for seg in raw_segments]

    sentences = _words_to_sentences(all_words, lang)

    if log:
        log.info(f"{lang}: {len(sentences)} sentences")

    return SegmentedText(sentences=sentences)


def _decode_to_wav(audio_path: Path) -> Path:
    """Decode audio to a temp 16kHz mono WAV for fast random-access energy reads."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1",
            "-f", "wav", "-acodec", "pcm_f32le",
            tmp_path,
        ],
        capture_output=True,
        check=True,
    )
    return Path(tmp_path)


def refine_timestamps(
    segmented: SegmentedText,
    audio_path: Path,
    log: PipelineLog | None = None,
    threshold: float = 0.005,
    max_extend_ms: int = 300,
    win_ms: int = 10,
) -> SegmentedText:
    """Refine segment end timestamps using energy-based speech boundary detection.

    Scans forward from each segment's end in small windows until RMS drops
    below threshold, extending the end to the actual speech boundary.
    """
    wav_path = _decode_to_wav(audio_path)
    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate
        total_frames = info.frames

        win_samples = int(sr * win_ms / 1000)
        max_extend_samples = int(sr * max_extend_ms / 1000)

        extended_count = 0
        extensions_ms: list[float] = []

        for seg in segmented.sentences:
            end_frame = int(seg.end * sr)
            scan_end = min(total_frames, end_frame + max_extend_samples)

            if end_frame >= total_frames or win_samples == 0:
                continue

            # Read the region we need to scan
            data, _ = sf.read(
                str(wav_path), start=end_frame,
                stop=scan_end, dtype="float32",
            )

            # Scan forward in windows
            found_silence = False
            extend_samples = 0
            for offset in range(0, len(data) - win_samples + 1, win_samples):
                window = data[offset:offset + win_samples]
                rms = float(np.sqrt(np.mean(window ** 2)))
                if rms < threshold:
                    extend_samples = offset
                    found_silence = True
                    break

            if found_silence and extend_samples > 0:
                extend_sec = extend_samples / sr
                new_end = round(seg.end + extend_sec, 3)
                seg.end = new_end
                if seg.words:
                    seg.words[-1].end = new_end
                extended_count += 1
                extensions_ms.append(extend_sec * 1000)

        if log:
            total = len(segmented.sentences)
            if extended_count > 0:
                avg = sum(extensions_ms) / len(extensions_ms)
                log.info(
                    f"Extended {extended_count}/{total} segment ends "
                    f"(avg +{avg:.0f}ms)"
                )
            else:
                log.info(f"Refined timestamps: 0/{total} segments extended")

    finally:
        wav_path.unlink(missing_ok=True)

    return segmented
