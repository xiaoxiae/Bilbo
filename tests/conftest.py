import pytest
from pathlib import Path

from bilbo.library import Library
from bilbo.models import Segment, SegmentedText, Alignment, AlignmentPair, BookMeta, Word


@pytest.fixture
def tmp_library(tmp_path):
    lib = Library(root=tmp_path / "bilbo_lib")
    lib.init()
    return lib


@pytest.fixture
def sample_segments():
    return [
        Segment(start=0.0, end=1.5, text="Hello world.", words=[
            Word(0.0, 0.7, "Hello"), Word(0.8, 1.5, "world."),
        ]),
        Segment(start=1.5, end=3.0, text="How are you?", words=[
            Word(1.5, 1.8, "How"), Word(1.9, 2.3, "are"), Word(2.4, 3.0, "you?"),
        ]),
        Segment(start=5.0, end=7.0, text="I am fine.", words=[
            Word(5.0, 5.3, "I"), Word(5.4, 5.8, "am"), Word(5.9, 7.0, "fine."),
        ]),
        Segment(start=7.0, end=9.0, text="Thank you.", words=[
            Word(7.0, 7.8, "Thank"), Word(7.9, 9.0, "you."),
        ]),
    ]


@pytest.fixture
def sample_segmented_text(sample_segments):
    return SegmentedText(sentences=sample_segments)


@pytest.fixture
def sample_alignment():
    return Alignment(pairs=[
        AlignmentPair(
            l1=[Segment(0.0, 1.5, "Hello world.", words=[
                Word(0.0, 0.7, "Hello"), Word(0.8, 1.5, "world."),
            ])],
            l2=[Segment(0.0, 1.8, "Hallo Welt.", words=[
                Word(0.0, 0.8, "Hallo"), Word(0.9, 1.8, "Welt."),
            ])],
        ),
        AlignmentPair(
            l1=[Segment(1.5, 3.0, "How are you?", words=[
                Word(1.5, 1.8, "How"), Word(1.9, 2.3, "are"), Word(2.4, 3.0, "you?"),
            ])],
            l2=[Segment(1.8, 3.5, "Wie geht es dir?", words=[
                Word(1.8, 2.1, "Wie"), Word(2.2, 2.6, "geht"), Word(2.7, 3.0, "es"), Word(3.1, 3.5, "dir?"),
            ])],
        ),
    ])


@pytest.fixture
def sample_meta():
    return BookMeta(
        slug="test-book",
        title="Test Book",
        l1_lang="en",
        l2_lang="de",
        l1_audio="/tmp/test_en.mp3",
        l2_audio="/tmp/test_de.mp3",
    )
