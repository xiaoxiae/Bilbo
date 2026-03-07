import numpy as np
import soundfile as sf

from bilbo.audio import AudioExporter, crossfade, generate_silence, slice_audio


def test_slice_audio(tmp_path):
    sr = 16000
    data = np.random.randn(sr * 10, 2).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), data, sr)

    chunk = slice_audio(wav_path, sr, 2.0, 3.0, padding_ms=0)
    expected_len = sr  # 1 second
    assert abs(len(chunk) - expected_len) < 2


def test_slice_audio_with_padding(tmp_path):
    sr = 16000
    data = np.random.randn(sr * 10, 2).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), data, sr)

    chunk = slice_audio(wav_path, sr, 2.0, 3.0, padding_ms=100)
    # 1 second + 200ms padding
    assert len(chunk) > sr


def test_generate_silence():
    silence = generate_silence(16000, 500, channels=2)
    assert silence.shape == (8000, 2)
    assert np.all(silence == 0)


def test_crossfade_basic():
    a = np.ones((1000, 2), dtype=np.float32)
    b = np.ones((1000, 2), dtype=np.float32) * 0.5
    result = crossfade(a, b, ms=10)
    assert len(result) < len(a) + len(b)
    assert len(result) > 0


def test_crossfade_empty():
    a = np.ones((1000, 2), dtype=np.float32)
    b = np.zeros((0, 2), dtype=np.float32)
    assert np.array_equal(crossfade(a, b), a)
    assert np.array_equal(crossfade(b, a), a)


def test_audio_exporter_streams_to_file(tmp_path):
    """Chunks written to AudioExporter produce a valid audio file."""
    sr = 16000
    out = tmp_path / "out.mp3"
    chunk = np.random.randn(sr, 1).astype(np.float32)  # 1 second mono

    with AudioExporter(sr, 1, out, fmt="mp3") as exp:
        exp.write(chunk)
        exp.write(chunk)

    assert exp.total_samples == sr * 2
    assert abs(exp.duration - 2.0) < 0.01
    assert out.exists()
    info = sf.info(str(out))
    assert abs(info.duration - 2.0) < 0.1


def test_audio_exporter_skips_empty_chunks(tmp_path):
    sr = 16000
    out = tmp_path / "out.mp3"
    chunk = np.random.randn(sr, 1).astype(np.float32)
    empty = np.zeros((0, 1), dtype=np.float32)

    with AudioExporter(sr, 1, out, fmt="mp3") as exp:
        exp.write(empty)
        exp.write(chunk)
        exp.write(empty)

    assert exp.total_samples == sr


def test_audio_exporter_progress_callback(tmp_path):
    sr = 16000
    out = tmp_path / "out.mp3"
    chunk = np.random.randn(sr * 3, 1).astype(np.float32)  # 3 seconds
    progress_calls = []

    with AudioExporter(sr, 1, out, fmt="mp3",
                       on_progress=lambda cur, tot: progress_calls.append((cur, tot))) as exp:
        exp.write(chunk)

    assert len(progress_calls) > 0
    # Each call should have (current_secs, total_secs)
    last_cur, last_tot = progress_calls[-1]
    assert last_cur > 0
    assert last_tot > 0
