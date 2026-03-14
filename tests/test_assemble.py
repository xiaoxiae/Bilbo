import numpy as np
import soundfile as sf

from bilbo.models import AlignmentPair, ExportConfig, Segment, Word


def test_export_config_defaults():
    c = ExportConfig()
    assert c.intra_gap_ms == 300
    assert c.inter_gap_ms == 600
    assert c.format == "m4b"
    assert c.fade_ms == 15


def test_extract_chunk_with_audio(tmp_path):
    """_extract_chunk reads the correct region from a WAV file."""
    from bilbo.assemble import _extract_chunk

    sr = 16000
    channels = 1
    duration_s = 2.0
    n_samples = int(sr * duration_s)
    data = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
    data = data.reshape(-1, 1)

    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), data, sr)

    pair = AlignmentPair(
        l1=[Segment(0.5, 1.0, "Hello.", words=[Word(0.5, 1.0, "Hello.")])],
        l2=[Segment(0.0, 0.5, "Hallo.", words=[Word(0.0, 0.5, "Hallo.")])],
    )
    config = ExportConfig(padding_ms=0, fade_ms=0)
    chunk = _extract_chunk(pair, wav_path, sr, "l1", config)
    # Should contain approximately 0.5s of audio
    assert len(chunk) > 0
    assert chunk.shape[1] == channels


def test_extract_chunk_empty_segments(tmp_path):
    """_extract_chunk with empty segments returns empty array."""
    from bilbo.assemble import _extract_chunk

    sr = 16000
    data = np.zeros((sr, 1), dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), data, sr)

    pair = AlignmentPair(l1=[], l2=[Segment(0.0, 0.5, "A.", words=[])])
    config = ExportConfig()
    chunk = _extract_chunk(pair, wav_path, sr, "l1", config)
    assert len(chunk) == 0


def test_build_text_meta_both_sources():
    """_build_text_meta merges titles from both L1 and L2."""
    from bilbo.assemble import _build_text_meta
    from bilbo.metadata import SourceMetadata

    l1 = SourceMetadata(title="English Book", artist="Author A")
    l2 = SourceMetadata(title="German Book", artist="Author B")
    config = ExportConfig(llm_merge=False)
    result = _build_text_meta(l1, l2, config, None)
    assert result["title"] == "English Book / German Book"
    assert result["artist"] == "Author A / Author B"


def test_build_text_meta_single_source():
    """_build_text_meta with only L1 metadata."""
    from bilbo.assemble import _build_text_meta
    from bilbo.metadata import SourceMetadata

    l1 = SourceMetadata(title="English Book", artist="Author A")
    l2 = SourceMetadata()
    config = ExportConfig(llm_merge=False)
    result = _build_text_meta(l1, l2, config, None)
    assert result["title"] == "English Book"
    assert result["artist"] == "Author A"


def test_build_text_meta_empty():
    """_build_text_meta with no metadata returns empty dict."""
    from bilbo.assemble import _build_text_meta
    from bilbo.metadata import SourceMetadata

    result = _build_text_meta(SourceMetadata(), SourceMetadata(), ExportConfig(), None)
    assert result == {}
