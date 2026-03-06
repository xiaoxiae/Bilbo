from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class SegmentedText:
    sentences: list[Segment]

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> SegmentedText:
        data = json.loads(path.read_text())
        return cls(
            sentences=[
                Segment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    words=[Word(**w) for w in s.get("words", [])],
                )
                for s in data["sentences"]
            ],
        )


@dataclass
class AlignmentPair:
    l1: list[Segment]
    l2: list[Segment]
    score: float = 0.0


@dataclass
class Alignment:
    pairs: list[AlignmentPair]

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        data = [{"l1": [asdict(s) for s in p.l1], "l2": [asdict(s) for s in p.l2], "score": p.score} for p in self.pairs]
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> Alignment:
        data = json.loads(path.read_text())
        pairs = [
            AlignmentPair(
                l1=[Segment(
                    start=s["start"], end=s["end"], text=s["text"],
                    words=[Word(**w) for w in s.get("words", [])],
                ) for s in p["l1"]],
                l2=[Segment(
                    start=s["start"], end=s["end"], text=s["text"],
                    words=[Word(**w) for w in s.get("words", [])],
                ) for s in p["l2"]],
                score=p.get("score", 0.0),
            )
            for p in data
        ]
        return cls(pairs=pairs)


@dataclass
class BookMeta:
    slug: str
    title: str
    l1_lang: str
    l2_lang: str
    l1_audio: str
    l2_audio: str
    stages_completed: list[int] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BookMeta:
        return cls(**d)


@dataclass
class ExportConfig:
    intra_gap_ms: int = 300
    inter_gap_ms: int = 600
    format: str = "m4b"
    order: str = "l1-first"
    crossfade_ms: int = 30
    padding_ms: int = 75
