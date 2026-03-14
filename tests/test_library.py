from bilbo.library import Library, _slugify
from bilbo.models import BookMeta


def test_library_init(tmp_library):
    assert tmp_library.root.exists()
    assert tmp_library.books_dir.exists()
    assert tmp_library.index_path.exists()


def test_slugify():
    assert _slugify("Test Book") == "test-book"
    assert _slugify("  Hello   World  ") == "hello-world"
    assert _slugify("Über cool!") == "über-cool"
    assert _slugify("---") == "book"
    assert _slugify("") == "book"


def test_add_and_get(tmp_library, sample_meta):
    tmp_library.add_or_update(sample_meta)
    retrieved = tmp_library.get("test-book")
    assert retrieved is not None
    assert retrieved.title == "Test Book"
    assert retrieved.slug == "test-book"
    assert retrieved.l1_lang == "en"


def test_list_books(tmp_library, sample_meta):
    assert tmp_library.list_books() == []
    tmp_library.add_or_update(sample_meta)
    books = tmp_library.list_books()
    assert len(books) == 1
    assert books[0].title == "Test Book"


def test_delete(tmp_library, sample_meta):
    tmp_library.add_or_update(sample_meta)
    assert tmp_library.delete("test-book")
    assert tmp_library.get("test-book") is None
    assert not tmp_library.delete("nonexistent")


def test_make_slug(tmp_library, sample_meta):
    assert tmp_library.make_slug("The Trial") == "the-trial"

    tmp_library.add_or_update(sample_meta)
    assert tmp_library.make_slug("Test Book") == "test-book-2"


def test_find_by_title(tmp_library, sample_meta):
    assert tmp_library.find_by_title("Test Book") is None
    tmp_library.add_or_update(sample_meta)
    found = tmp_library.find_by_title("Test Book")
    assert found is not None
    assert found.slug == "test-book"
    assert tmp_library.find_by_title("Other") is None


def test_rename(tmp_library, sample_meta):
    tmp_library.add_or_update(sample_meta)
    assert tmp_library.book_dir("test-book").exists()

    result = tmp_library.rename("Test Book", "New Title")
    assert result is not None
    assert result.title == "New Title"
    assert result.slug == "new-title"
    assert tmp_library.get("test-book") is None
    assert tmp_library.get("new-title") is not None
    assert tmp_library.book_dir("new-title").exists()
    assert not tmp_library.book_dir("test-book").exists()


def test_rename_not_found(tmp_library):
    assert tmp_library.rename("nonexistent", "Whatever") is None


def test_rename_slug_in_parent_path(tmp_path):
    """Rename should not corrupt paths when slug appears in parent directories."""
    # Create library where the root path contains the slug as a substring
    lib_root = tmp_path / "test-book" / "library"
    lib = Library(root=lib_root)
    lib.init()

    book_dir = lib.book_dir("test-book")
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "exports").mkdir(exist_ok=True)
    input_dir = book_dir / "input"
    input_dir.mkdir(exist_ok=True)

    meta = BookMeta(
        slug="test-book",
        title="Test Book",
        l1_lang="en",
        l2_lang="de",
        l1_audio=str(input_dir / "l1.mp3"),
        l2_audio=str(input_dir / "l2.mp3"),
    )
    lib.add_or_update(meta)

    result = lib.rename("Test Book", "New Title")
    assert result is not None
    # The parent "test-book" directory in the path should NOT be renamed
    new_dir = lib.book_dir("new-title")
    assert "test-book" in str(new_dir.parent)  # parent still has "test-book"
    assert result.l1_audio.endswith("new-title/input/l1.mp3")
    assert result.l2_audio.endswith("new-title/input/l2.mp3")


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
