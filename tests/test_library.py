import pytest

from bilbo.library import Library, _slugify
from bilbo.models import BookMeta


def test_slugify():
    assert _slugify("The Trial") == "the-trial"
    assert _slugify("  Hello World!  ") == "hello-world"
    assert _slugify("Über den Wolken") == "über-den-wolken"


def test_library_init(tmp_library):
    assert tmp_library.root.exists()
    assert tmp_library.books_dir.exists()
    assert tmp_library.index_path.exists()


def test_add_and_get(tmp_library, sample_meta):
    tmp_library.add_or_update(sample_meta)
    retrieved = tmp_library.get("test-book")
    assert retrieved is not None
    assert retrieved.title == "Test Book"
    assert retrieved.l1_lang == "en"


def test_list_books(tmp_library, sample_meta):
    assert tmp_library.list_books() == []
    tmp_library.add_or_update(sample_meta)
    books = tmp_library.list_books()
    assert len(books) == 1
    assert books[0].slug == "test-book"


def test_delete(tmp_library, sample_meta):
    tmp_library.add_or_update(sample_meta)
    assert tmp_library.delete("test-book")
    assert tmp_library.get("test-book") is None
    assert not tmp_library.delete("nonexistent")


def test_make_slug(tmp_library, sample_meta):
    slug = tmp_library.make_slug("The Trial")
    assert slug == "the-trial"

    tmp_library.add_or_update(sample_meta)
    slug2 = tmp_library.make_slug("Test Book")
    assert slug2 == "test-book-2"


def test_model_serialization(tmp_path):
    from bilbo.models import SegmentedText, Segment, Alignment, AlignmentPair

    seg = SegmentedText(
        sentences=[Segment(0.0, 1.0, "Hello."), Segment(1.0, 2.0, "World.")],
    )
    path = tmp_path / "seg.json"
    seg.save(path)
    loaded = SegmentedText.load(path)
    assert len(loaded.sentences) == 2
    assert loaded.sentences[0].text == "Hello."

    alignment = Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(0, 1, "Hi")],
            l2=[Segment(0, 1, "Hallo")],
        )
    ])
    apath = tmp_path / "align.json"
    alignment.save(apath)
    loaded_a = Alignment.load(apath)
    assert len(loaded_a.pairs) == 1
    assert loaded_a.pairs[0].l1[0].text == "Hi"
