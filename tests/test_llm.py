from unittest.mock import patch, MagicMock
import json

from bilbo.llm import (
    is_available,
    merge_metadata_text,
    merge_chapter_titles,
    _simple_merge,
)


def test_simple_merge_identical():
    assert _simple_merge("Author A", "Author A") == "Author A"


def test_simple_merge_different():
    assert _simple_merge("English", "German") == "English / German"


def test_is_available_unreachable():
    with patch("bilbo.llm.urllib.request.urlopen", side_effect=OSError):
        assert is_available() is False


def test_is_available_reachable():
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("bilbo.llm.urllib.request.urlopen", return_value=mock_resp):
        assert is_available() is True


def test_merge_metadata_fallback_on_error():
    """When ollama is unreachable, falls back to simple merging."""
    with patch("bilbo.llm._generate", side_effect=OSError("connection refused")):
        result = merge_metadata_text({
            "title": ("The Book", "Das Buch"),
            "artist": ("Author", "Author"),
        })
    assert result["title"] == "The Book / Das Buch"
    assert result["artist"] == "Author"


def test_merge_metadata_empty():
    assert merge_metadata_text({}) == {}


def test_merge_metadata_success():
    response = json.dumps({"title": "The Book / Das Buch", "artist": "Same Author"})
    with patch("bilbo.llm._generate", return_value=response):
        result = merge_metadata_text({
            "title": ("The Book", "Das Buch"),
            "artist": ("Same Author", "Same Author"),
        })
    assert result["title"] == "The Book / Das Buch"
    assert result["artist"] == "Same Author"


def test_merge_metadata_partial_response():
    """If LLM returns only some keys, fallback fills the rest."""
    response = json.dumps({"title": "Merged Title"})
    with patch("bilbo.llm._generate", return_value=response):
        result = merge_metadata_text({
            "title": ("Book A", "Book B"),
            "artist": ("Author A", "Author B"),
        })
    assert result["title"] == "Merged Title"
    assert result["artist"] == "Author A / Author B"


def test_merge_metadata_invalid_json():
    with patch("bilbo.llm._generate", return_value="not json at all"):
        result = merge_metadata_text({
            "title": ("A", "B"),
        })
    assert result["title"] == "A / B"


def test_merge_chapters_fallback_on_error():
    with patch("bilbo.llm._generate", side_effect=OSError):
        result = merge_chapter_titles([
            ("Chapter 1", ["Kapitel 1"]),
            ("Chapter 2", ["Track 1", "Track 2"]),
        ])
    assert result == ["Chapter 1 / Kapitel 1", "Chapter 2 / Track 1, Track 2"]


def test_merge_chapters_empty():
    assert merge_chapter_titles([]) == []


def test_merge_chapters_success():
    response = json.dumps({"t0": "Chapter 1", "t1": "Chapter 2 (Tracks 1-2)"})
    with patch("bilbo.llm._generate", return_value=response):
        result = merge_chapter_titles([
            ("Chapter 1", ["Kapitel 1"]),
            ("Chapter 2", ["Track 1", "Track 2"]),
        ])
    assert result == ["Chapter 1", "Chapter 2 (Tracks 1-2)"]


def test_merge_chapters_missing_key():
    """If LLM omits a key, fall back for that entry."""
    response = json.dumps({"t0": "Ch 1"})
    with patch("bilbo.llm._generate", return_value=response):
        result = merge_chapter_titles([
            ("Ch 1", ["K 1"]),
            ("Ch 2", ["K 2"]),
        ])
    assert result[0] == "Ch 1"
    assert result[1] == "Ch 2 / K 2"


def test_merge_chapters_identical_titles():
    """Identical L1/L2 should deduplicate in fallback."""
    with patch("bilbo.llm._generate", side_effect=OSError):
        result = merge_chapter_titles([
            ("Chapter 1", ["Chapter 1"]),
        ])
    assert result == ["Chapter 1"]


def test_merge_chapters_no_l2():
    """Entry with no L2 titles uses L1 only."""
    with patch("bilbo.llm._generate", side_effect=OSError):
        result = merge_chapter_titles([
            ("Chapter 1", []),
        ])
    assert result == ["Chapter 1"]
