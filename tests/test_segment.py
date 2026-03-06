from bilbo.models import Segment, Word
from bilbo.segment import _words_to_sentences, segment_text


def test_words_to_sentences_basic():
    words = [
        Word(0.0, 0.5, "Hello"),
        Word(0.6, 1.0, "world."),
        Word(1.5, 1.8, "How"),
        Word(1.9, 2.3, "are"),
        Word(2.4, 3.0, "you?"),
    ]
    result = _words_to_sentences(words, "en")
    assert len(result) >= 2
    assert result[0].text == "Hello world."
    assert result[0].start == 0.0
    assert result[0].end == 1.0


def test_words_to_sentences_empty():
    assert _words_to_sentences([], "en") == []


def test_segment_text_basic():
    segs = [
        Segment(start=0.0, end=2.0, text="Hello world.", words=[
            Word(0.0, 0.5, "Hello"), Word(0.6, 2.0, "world."),
        ]),
        Segment(start=2.0, end=4.0, text="How are you?", words=[
            Word(2.0, 2.3, "How"), Word(2.4, 2.8, "are"), Word(2.9, 4.0, "you?"),
        ]),
    ]
    result = segment_text(segs, "en")
    assert len(result.sentences) >= 1


def test_segment_text_no_words_fallback():
    segs = [
        Segment(start=0.0, end=2.0, text="Hello world."),
        Segment(start=2.0, end=4.0, text="How are you?"),
    ]
    result = segment_text(segs, "en")
    assert len(result.sentences) >= 1
