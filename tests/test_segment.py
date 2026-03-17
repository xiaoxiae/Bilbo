import numpy as np
import pytest
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


def test_refine_trims_trailing_silence(tmp_path):
    """VAD speech ends before seg.end → trim to speech end + 50ms padding."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.8, text="hello", words=[Word(0.0, 0.8, "hello")]),
        Segment(start=1.0, end=1.5, text="world", words=[Word(1.0, 1.5, "world")]),
    ])
    vad = [{"start": 0.0, "end": 0.5}, {"start": 1.0, "end": 1.5}]

    refined, stats = refine_timestamps(segs, wav, _vad_result=vad)
    assert refined.sentences[0].end == pytest.approx(0.55)
    assert refined.sentences[0].words[-1].end == pytest.approx(0.55)
    assert refined.sentences[0].start == pytest.approx(0.0)  # clamped at prev_end=0
    assert stats["contracted"] == 1


def test_refine_extends_to_actual_speech_end(tmp_path):
    """VAD speech ends after seg.end → extend with 50ms padding."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.4, text="hello", words=[Word(0.0, 0.4, "hello")]),
        Segment(start=1.0, end=1.5, text="world", words=[Word(1.0, 1.5, "world")]),
    ])
    # Only provide VAD for the first segment so the count is unambiguous
    vad = [{"start": 0.0, "end": 0.5}]

    refined, stats = refine_timestamps(segs, wav, _vad_result=vad)
    assert refined.sentences[0].end == pytest.approx(0.55)
    assert refined.sentences[0].words[-1].end == pytest.approx(0.55)
    assert stats["extended"] == 1


def test_refine_clamps_at_next_start(tmp_path):
    """VAD speech spans into next segment → clamp at next_seg.start."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.8, text="hello", words=[Word(0.0, 0.8, "hello")]),
        Segment(start=1.0, end=1.5, text="world", words=[Word(1.0, 1.5, "world")]),
    ])
    # VAD region spans across boundary; 0.8/1.2 = 67% inside seg[0] → matched
    vad = [{"start": 0.0, "end": 1.2}]

    refined, _ = refine_timestamps(segs, wav, _vad_result=vad)
    assert refined.sentences[0].end == 1.0  # clamped to next_seg.start


def test_refine_also_refines_start(tmp_path):
    """VAD region matched → start also snaps to vad_start - 50ms padding."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.1, end=0.8, text="hello", words=[Word(0.1, 0.8, "hello")]),
        Segment(start=1.0, end=1.5, text="world", words=[Word(1.0, 1.5, "world")]),
    ])
    # VAD[0.2,0.6]: overlap with seg[0.1,0.8] = 0.4, dur=0.4, frac=1.0 → matched
    vad = [{"start": 0.2, "end": 0.6}, {"start": 1.0, "end": 1.5}]

    refined, _ = refine_timestamps(segs, wav, _vad_result=vad)
    assert refined.sentences[0].start == pytest.approx(0.15)  # 0.2 - 0.05, clamped at 0.0
    assert refined.sentences[0].words[0].start == pytest.approx(0.15)


def test_refine_no_match_leaves_unchanged(tmp_path):
    """VAD region overlaps < 50% of its duration with segment → no change."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.5, text="hello", words=[Word(0.0, 0.5, "hello")]),
    ])
    # VAD[0.4,0.9]: overlap with seg[0,0.5] = 0.1, dur=0.5, frac=0.2 < 0.5 → no match
    vad = [{"start": 0.4, "end": 0.9}]

    refined, stats = refine_timestamps(segs, wav, _vad_result=vad)
    assert refined.sentences[0].start == 0.0
    assert refined.sentences[0].end == 0.5
    assert stats["adjusted"] == 0


def test_refine_last_segment_uses_vad_end(tmp_path):
    """Last segment has no upper bound → end = vad_end + 50ms padding."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.5, text="hello", words=[Word(0.0, 0.5, "hello")]),
        Segment(start=1.0, end=1.8, text="world", words=[Word(1.0, 1.8, "world")]),
    ])
    vad = [{"start": 0.0, "end": 0.4}, {"start": 1.0, "end": 1.5}]

    refined, _ = refine_timestamps(segs, wav, _vad_result=vad)
    assert refined.sentences[1].end == pytest.approx(1.55)


def test_refine_no_vad_regions_no_change(tmp_path):
    """Empty VAD result → all endpoints unchanged."""
    wav = _make_wav(tmp_path, 2.0, [])
    segs = SegmentedText(sentences=[
        Segment(start=0.0, end=0.5, text="hello", words=[Word(0.0, 0.5, "hello")]),
        Segment(start=1.0, end=1.5, text="world", words=[Word(1.0, 1.5, "world")]),
    ])

    refined, stats = refine_timestamps(segs, wav, _vad_result=[])
    assert refined.sentences[0].end == 0.5
    assert refined.sentences[1].end == 1.5
    assert stats["adjusted"] == 0


def test_refine_stats_correctness(tmp_path):
    """Stats dict has correct structure and values."""
    wav = _make_wav(tmp_path, 3.0, [])
    segs = SegmentedText(sentences=[
        # end=0.4, VAD=[0,0.5] → new_end=0.55, extended
        Segment(start=0.0, end=0.4, text="hello", words=[Word(0.0, 0.4, "hello")]),
        # end=1.8, VAD=[1.0,1.5] → new_end=1.55, contracted
        Segment(start=1.0, end=1.8, text="world", words=[Word(1.0, 1.8, "world")]),
        # end=2.5, VAD=[2.0,2.45] → new_end=2.5, end unchanged
        Segment(start=2.0, end=2.5, text="bye", words=[Word(2.0, 2.5, "bye")]),
    ])
    vad = [{"start": 0.0, "end": 0.5}, {"start": 1.0, "end": 1.5}, {"start": 2.0, "end": 2.45}]

    _, stats = refine_timestamps(segs, wav, _vad_result=vad)
    assert stats["total"] == 3
    assert stats["adjusted"] == 2
    assert stats["extended"] == 1
    assert stats["contracted"] == 1
    assert stats["avg_extend_ms"] > 0
    assert stats["avg_contract_ms"] > 0
    assert set(stats.keys()) == {"adjusted", "extended", "contracted", "total", "avg_extend_ms", "avg_contract_ms"}
