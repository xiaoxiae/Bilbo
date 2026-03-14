import subprocess

import numpy as np
import pytest
import soundfile as sf

from bilbo.models import Segment, SegmentedText, Word
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
    assert len(result) == 2
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
    assert len(result.sentences) == 2


def test_segment_text_no_words_fallback():
    segs = [
        Segment(start=0.0, end=2.0, text="Hello world."),
        Segment(start=2.0, end=4.0, text="How are you?"),
    ]
    result = segment_text(segs, "en")
    assert len(result.sentences) == 2


def _has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_refine_timestamps_extends_speech(tmp_path):
    """Tone extending beyond segment end should cause extension."""
    from bilbo.segment import refine_timestamps

    sr = 16000
    duration_s = 2.0
    n_samples = int(sr * duration_s)
    # Continuous tone for first 1.5s, then silence
    t = np.arange(n_samples, dtype=np.float32) / sr
    data = np.where(t < 1.5, np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5, 0.0).astype(np.float32)

    wav_path = tmp_path / "audio.wav"
    sf.write(str(wav_path), data, sr)

    seg = SegmentedText(sentences=[
        Segment(start=0.0, end=1.0, text="Hello.", words=[Word(0.0, 1.0, "Hello.")]),
    ])
    result, stats = refine_timestamps(seg, wav_path, max_extend_ms=1000)
    # Segment end should be extended because tone continues past 1.0s
    assert result.sentences[0].end > 1.0
    assert stats["extended"] == 1
    assert stats["total"] == 1
    assert stats["avg_ms"] > 0


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_refine_timestamps_no_extension_silence(tmp_path):
    """Segment end followed immediately by silence should not extend."""
    from bilbo.segment import refine_timestamps

    sr = 16000
    duration_s = 2.0
    n_samples = int(sr * duration_s)
    # Tone for first 0.5s, then silence
    t = np.arange(n_samples, dtype=np.float32) / sr
    data = np.where(t < 0.5, np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5, 0.0).astype(np.float32)

    wav_path = tmp_path / "audio.wav"
    sf.write(str(wav_path), data, sr)

    seg = SegmentedText(sentences=[
        Segment(start=0.0, end=0.5, text="Hi.", words=[Word(0.0, 0.5, "Hi.")]),
    ])
    result, stats = refine_timestamps(seg, wav_path)
    assert stats["extended"] == 0
    assert result.sentences[0].end == 0.5


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_refine_timestamps_stats_structure(tmp_path):
    """Stats dict should contain expected keys."""
    from bilbo.segment import refine_timestamps

    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    wav_path = tmp_path / "audio.wav"
    sf.write(str(wav_path), data, sr)

    seg = SegmentedText(sentences=[
        Segment(start=0.0, end=0.3, text="A.", words=[Word(0.0, 0.3, "A.")]),
    ])
    _, stats = refine_timestamps(seg, wav_path)
    assert "extended" in stats
    assert "total" in stats
    assert "avg_ms" in stats
    assert isinstance(stats["extended"], int)
    assert isinstance(stats["total"], int)
    assert isinstance(stats["avg_ms"], float)
