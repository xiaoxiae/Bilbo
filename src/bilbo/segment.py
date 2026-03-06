from __future__ import annotations

import click
import pysbd

from .models import Segment, SegmentedText, Word


def _words_to_sentences(
    words: list[Word], lang: str
) -> list[Segment]:
    if not words:
        return []

    full_text = " ".join(w.word for w in words)

    segmenter = pysbd.Segmenter(language=lang, clean=False)
    sentence_texts = segmenter.segment(full_text)
    sentence_texts = [s.strip() for s in sentence_texts if s.strip()]

    if not sentence_texts:
        return []

    # Build a mapping from character position in full_text to word index
    # Each word contributes its text + a space separator
    char_to_word: list[int] = []
    for wi, w in enumerate(words):
        char_to_word.extend([wi] * len(w.word))
        if wi < len(words) - 1:
            char_to_word.append(wi)  # space between words

    sentences = []
    search_from = 0

    for sent_text in sentence_texts:
        if not sent_text.strip():
            continue

        idx = full_text.find(sent_text, search_from)
        if idx == -1:
            idx = search_from

        sent_end_char = min(idx + len(sent_text) - 1, len(char_to_word) - 1)
        idx = min(idx, len(char_to_word) - 1)

        if idx < 0 or sent_end_char < 0:
            continue

        first_word_idx = char_to_word[idx]
        last_word_idx = char_to_word[sent_end_char]
        search_from = idx + len(sent_text)

        start = words[first_word_idx].start
        end = words[last_word_idx].end
        sent_segment_words = words[first_word_idx:last_word_idx + 1]
        sentences.append(Segment(
            start=round(start, 3),
            end=round(end, 3),
            text=sent_text,
            words=sent_segment_words,
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
