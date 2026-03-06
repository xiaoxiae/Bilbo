import numpy as np

from bilbo.audio import crossfade, generate_silence, normalize_lufs, slice_audio


def test_slice_audio():
    sr = 16000
    data = np.random.randn(sr * 10, 2)  # 10 seconds
    chunk = slice_audio(data, sr, 2.0, 3.0, padding_ms=0)
    expected_len = sr  # 1 second
    assert abs(len(chunk) - expected_len) < 2


def test_slice_audio_with_padding():
    sr = 16000
    data = np.random.randn(sr * 10, 2)
    chunk = slice_audio(data, sr, 2.0, 3.0, padding_ms=100)
    # 1 second + 200ms padding
    assert len(chunk) > sr


def test_generate_silence():
    silence = generate_silence(16000, 500, channels=2)
    assert silence.shape == (8000, 2)
    assert np.all(silence == 0)


def test_crossfade_basic():
    a = np.ones((1000, 2))
    b = np.ones((1000, 2)) * 0.5
    result = crossfade(a, b, ms=10)
    assert len(result) < len(a) + len(b)
    assert len(result) > 0


def test_crossfade_empty():
    a = np.ones((1000, 2))
    b = np.zeros((0, 2))
    assert np.array_equal(crossfade(a, b), a)
    assert np.array_equal(crossfade(b, a), a)


def test_normalize_lufs():
    sr = 44100
    data = np.random.randn(sr * 2, 2) * 0.1
    result = normalize_lufs(data, sr)
    assert result.shape == data.shape
