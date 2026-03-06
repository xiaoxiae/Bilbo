from __future__ import annotations

from typing import TYPE_CHECKING

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
