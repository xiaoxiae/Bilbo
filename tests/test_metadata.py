from bilbo.metadata import (
    SourceChapter,
    SourceMetadata,
    map_chapters_to_output,
    save_source_metadata,
    load_source_metadata,
)
from bilbo.models import (
    Alignment,
    AlignmentPair,
    ChapterMarker,
    Segment,
    Word,
)


def test_source_metadata_roundtrip(tmp_path):
    l1 = SourceMetadata(
        title="English Book",
        artist="Author A",
        chapters=[SourceChapter("Ch1", 0.0, 60.0), SourceChapter("Ch2", 60.0, 120.0)],
        has_cover=True,
    )
    l2 = SourceMetadata(
        title="German Book",
        artist="Author B",
        chapters=[SourceChapter("Kap1", 0.0, 55.0), SourceChapter("Kap2", 55.0, 110.0)],
        has_cover=False,
    )
    path = tmp_path / "meta.json"
    save_source_metadata(l1, l2, path)
    loaded_l1, loaded_l2 = load_source_metadata(path)

    assert loaded_l1.title == "English Book"
    assert loaded_l1.artist == "Author A"
    assert len(loaded_l1.chapters) == 2
    assert loaded_l1.chapters[0].title == "Ch1"
    assert loaded_l1.has_cover is True

    assert loaded_l2.title == "German Book"
    assert loaded_l2.has_cover is False


def test_map_chapters_simple():
    """Two source chapters, two alignment pairs — one per chapter."""
    l1_chapters = [
        SourceChapter("Chapter 1", 0.0, 5.0),
        SourceChapter("Chapter 2", 5.0, 10.0),
    ]
    l2_chapters = [
        SourceChapter("Kapitel 1", 0.0, 4.5),
        SourceChapter("Kapitel 2", 4.5, 9.0),
    ]
    alignment = Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(1.0, 4.0, "Hello.", words=[Word(1.0, 2.0, "Hello.")])],
            l2=[Segment(0.5, 3.5, "Hallo.", words=[Word(0.5, 1.5, "Hallo.")])],
        ),
        AlignmentPair(
            l1=[Segment(6.0, 9.0, "World.", words=[Word(6.0, 7.0, "World.")])],
            l2=[Segment(5.0, 8.0, "Welt.", words=[Word(5.0, 6.0, "Welt.")])],
        ),
    ])
    pair_offsets_ms = [(0, 5000), (5000, 10000)]

    markers = map_chapters_to_output(l1_chapters, l2_chapters, alignment, pair_offsets_ms)

    assert len(markers) == 2
    assert markers[0].title == "Chapter 1 / Kapitel 1"
    assert markers[0].start_ms == 0
    assert markers[0].end_ms == 5000
    assert markers[1].title == "Chapter 2 / Kapitel 2"
    assert markers[1].start_ms == 5000
    assert markers[1].end_ms == 10000


def test_map_chapters_multiple_pairs_per_chapter():
    """Multiple alignment pairs map to the same chapter."""
    l1_chapters = [SourceChapter("Only Chapter", 0.0, 20.0)]
    l2_chapters = [SourceChapter("Einziges Kapitel", 0.0, 18.0)]
    alignment = Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(1.0, 3.0, "A.", words=[])],
            l2=[Segment(0.5, 2.5, "A.", words=[])],
        ),
        AlignmentPair(
            l1=[Segment(5.0, 8.0, "B.", words=[])],
            l2=[Segment(4.0, 7.0, "B.", words=[])],
        ),
    ])
    pair_offsets_ms = [(0, 3000), (3000, 8000)]

    markers = map_chapters_to_output(l1_chapters, l2_chapters, alignment, pair_offsets_ms)

    assert len(markers) == 1
    assert markers[0].title == "Only Chapter / Einziges Kapitel"
    assert markers[0].start_ms == 0
    assert markers[0].end_ms == 8000


def test_map_chapters_empty():
    """No chapters produces no markers."""
    alignment = Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(0.0, 1.0, "A.", words=[])],
            l2=[Segment(0.0, 1.0, "A.", words=[])],
        ),
    ])
    markers = map_chapters_to_output([], [], alignment, [(0, 1000)])
    assert markers == []


def test_map_chapters_l1_only():
    """L1 chapters only — titles are L1 only."""
    l1_chapters = [SourceChapter("Chapter 1", 0.0, 10.0)]
    alignment = Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(1.0, 5.0, "A.", words=[])],
            l2=[Segment(0.5, 4.5, "A.", words=[])],
        ),
    ])
    markers = map_chapters_to_output(l1_chapters, [], alignment, [(0, 5000)])
    assert len(markers) == 1
    assert markers[0].title == "Chapter 1"


def test_map_chapters_mismatched_counts():
    """L2 has more chapters than L1 — L2 titles matched by alignment, not index."""
    l1_chapters = [
        SourceChapter("Chapter 1", 0.0, 10.0),
        SourceChapter("Chapter 2", 10.0, 20.0),
    ]
    # L2 has 4 fine-grained chapters (e.g. CD tracks)
    l2_chapters = [
        SourceChapter("Track 1", 0.0, 2.5),
        SourceChapter("Track 2", 2.5, 5.0),
        SourceChapter("Track 3", 5.0, 7.5),
        SourceChapter("Track 4", 7.5, 10.0),
    ]
    alignment = Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(1.0, 4.0, "A.", words=[])],
            l2=[Segment(1.0, 2.0, "A.", words=[])],  # midpoint 1.5 -> Track 1
        ),
        AlignmentPair(
            l1=[Segment(5.0, 9.0, "B.", words=[])],
            l2=[Segment(3.0, 4.5, "B.", words=[])],  # midpoint 3.75 -> Track 2
        ),
        AlignmentPair(
            l1=[Segment(11.0, 15.0, "C.", words=[])],
            l2=[Segment(5.5, 7.0, "C.", words=[])],  # midpoint 6.25 -> Track 3
        ),
        AlignmentPair(
            l1=[Segment(16.0, 19.0, "D.", words=[])],
            l2=[Segment(8.0, 9.5, "D.", words=[])],  # midpoint 8.75 -> Track 4
        ),
    ])
    pair_offsets_ms = [(0, 3000), (3000, 6000), (6000, 9000), (9000, 12000)]

    markers = map_chapters_to_output(l1_chapters, l2_chapters, alignment, pair_offsets_ms)

    assert len(markers) == 2
    # Chapter 1 contains pairs 0,1 -> L2 Track 1, Track 2
    assert markers[0].title == "Chapter 1 / Track 1, Track 2"
    # Chapter 2 contains pairs 2,3 -> L2 Track 3, Track 4
    assert markers[1].title == "Chapter 2 / Track 3, Track 4"


def test_chapter_marker_fields():
    ch = ChapterMarker(title="Test", start_ms=0, end_ms=1000)
    assert ch.title == "Test"
    assert ch.start_ms == 0
    assert ch.end_ms == 1000


def test_source_metadata_from_dict_defaults():
    meta = SourceMetadata.from_dict({})
    assert meta.title is None
    assert meta.artist is None
    assert meta.chapters == []
    assert meta.has_cover is False
