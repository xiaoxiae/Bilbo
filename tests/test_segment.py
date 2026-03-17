import numpy as np
import soundfile as sf

from bilbo.models import Segment, SegmentedText, Word
from bilbo.segment import _words_to_sentences, refine_timestamps, segment_text


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


def _make_wav(tmp_path, duration_s, regions):
    """Create a synthetic 16kHz mono WAV with speech/silence regions.

    ``regions`` is a list of (start_s, end_s, amplitude) tuples.
    Unspecified regions default to silence (0.0).
    """
    sr = 16000
    n = int(duration_s * sr)
    samples = np.zeros(n, dtype=np.float32)
    for start_s, end_s, amp in regions:
        s = int(start_s * sr)
        e = int(end_s * sr)
        # Use noise so RMS ≈ amplitude
        samples[s:e] = amp * np.random.default_rng(42).standard_normal(e - s).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), samples, sr)
    return path


def test_refine_forward_extension(tmp_path):
    """Segment end is mid-speech — should extend forward to silence boundary."""
    # Speech from 0.0–0.5, silence from 0.5 onward.
    # Segment end at 0.4 (mid-speech) should extend forward to ~0.5.
    wav = _make_wav(tmp_path, 1.0, [(0.0, 0.5, 0.5)])
    seg = Segment(start=0.0, end=0.4, text="hello", words=[Word(0.0, 0.4, "hello")])
    st = SegmentedText(sentences=[seg])

    refined, stats = refine_timestamps(st, wav, win_ms=30, threshold=0.01)
    assert stats["adjusted"] == 1
    assert stats["extended"] == 1
    assert stats["contracted"] == 0
    # End should have moved forward (extended)
    assert refined.sentences[0].end > 0.4
    assert refined.sentences[0].end <= 0.55  # roughly at the silence boundary


def test_refine_backward_contraction(tmp_path):
    """Segment end is past speech in silence — should contract backward."""
    # Speech from 0.0–0.3, silence from 0.3 onward.
    # Segment end at 0.5 (well into silence) should contract back to ~0.3.
    wav = _make_wav(tmp_path, 1.0, [(0.0, 0.3, 0.5)])
    seg = Segment(start=0.0, end=0.5, text="hello", words=[Word(0.0, 0.5, "hello")])
    st = SegmentedText(sentences=[seg])

    refined, stats = refine_timestamps(st, wav, win_ms=30, threshold=0.01)
    assert stats["adjusted"] == 1
    assert stats["contracted"] == 1
    assert stats["extended"] == 0
    # End should have moved backward (contracted)
    assert refined.sentences[0].end < 0.5
    assert refined.sentences[0].end >= 0.25  # roughly at the speech boundary


def test_refine_bidirectional_picks_closer(tmp_path):
    """When both directions find silence, pick the closer boundary."""
    # Speech 0.0–0.3, silence 0.3–0.4, speech 0.4–0.7, silence 0.7+
    # Segment end at 0.35 (in the silence gap). Backward finds ~0.3 (dist 0.05),
    # forward finds ~0.4... but 0.4 is speech so forward scan continues to 0.7.
    # Backward at ~0.3 is closer, so it should contract.
    wav = _make_wav(tmp_path, 1.0, [(0.0, 0.3, 0.5), (0.4, 0.7, 0.5)])
    seg = Segment(start=0.0, end=0.35, text="hello", words=[Word(0.0, 0.35, "hello")])
    st = SegmentedText(sentences=[seg])

    refined, stats = refine_timestamps(st, wav, win_ms=30, threshold=0.01)
    assert stats["adjusted"] == 1
    # Should have contracted (backward boundary was closer)
    assert refined.sentences[0].end < 0.35


def test_refine_no_change_when_at_silence(tmp_path):
    """No adjustment when segment end is already at silence boundary."""
    # Speech 0.0–0.5, silence 0.5+. Segment end exactly at 0.5.
    # Both backward (first window at 0.47–0.5 is speech) and forward (0.5–0.53
    # is silence) find boundaries. Forward is right at 0.5 (distance 0),
    # so effectively no change.
    wav = _make_wav(tmp_path, 1.0, [(0.0, 0.5, 0.5)])
    seg = Segment(start=0.0, end=0.5, text="hello", words=[Word(0.0, 0.5, "hello")])
    st = SegmentedText(sentences=[seg])

    refined, stats = refine_timestamps(st, wav, win_ms=30, threshold=0.01)
    # The forward scan finds silence immediately, so adjustment is 0 or tiny
    assert abs(refined.sentences[0].end - 0.5) <= 0.03


def test_refine_stats_correctness(tmp_path):
    """Stats dict has correct structure and values for multiple segments."""
    # Two segments: one will extend, one will contract.
    # Seg1: speech 0.0–0.5, end at 0.4 → extend forward
    # Seg2: speech 0.6–0.8, end at 0.95 → contract backward
    wav = _make_wav(tmp_path, 1.5, [(0.0, 0.5, 0.5), (0.6, 0.8, 0.5)])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.4, text="hello", words=[Word(0.0, 0.4, "hello")]),
        Segment(start=0.6, end=0.95, text="world", words=[Word(0.6, 0.95, "world")]),
    ])

    _, stats = refine_timestamps(segs, wav, win_ms=30, threshold=0.01)
    assert stats["total"] == 2
    assert stats["adjusted"] == 2
    assert stats["extended"] + stats["contracted"] == 2
    assert stats["avg_extend_ms"] >= 0
    assert stats["avg_contract_ms"] >= 0
    assert set(stats.keys()) == {"adjusted", "extended", "contracted", "total", "avg_extend_ms", "avg_contract_ms"}
