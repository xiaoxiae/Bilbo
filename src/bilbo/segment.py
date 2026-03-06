from __future__ import annotations

import click
import pysbd

from .models import Segment, SegmentedText, Word


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

    segmenter = pysbd.Segmenter(language=lang, clean=False, char_span=True)
    spans = segmenter.segment(full_text)

    sentences: list[Segment] = []
    wi = 0  # forward word pointer

    for span in spans:
        text = span.sent.strip()
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
) -> SegmentedText:
    click.echo(f"  Segmenting into sentences ({lang})...")

    # Flatten all words from raw segments
    all_words = []
    for seg in raw_segments:
        all_words.extend(seg.words)

    if not all_words:
        # Fallback: if no word-level timestamps, create one word per segment
        click.echo("  Warning: no word-level timestamps, using segment-level fallback")
        all_words = [Word(start=seg.start, end=seg.end, word=seg.text) for seg in raw_segments]

    sentences = _words_to_sentences(all_words, lang)

    click.echo(f"  {len(sentences)} sentences")

    return SegmentedText(sentences=sentences)
