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
    problematic_regions: list[tuple[int, int]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        obj = {
            "pairs": [{"l1": [asdict(s) for s in p.l1], "l2": [asdict(s) for s in p.l2], "score": p.score} for p in self.pairs],
            "problematic_regions": [list(r) for r in self.problematic_regions],
        }
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> Alignment:
        raw = json.loads(path.read_text())
        # Support old format (bare list of pairs) and new format (dict with pairs + regions)
        if isinstance(raw, list):
            pair_data = raw
            regions = []
        else:
            pair_data = raw["pairs"]
            regions = [tuple(r) for r in raw.get("problematic_regions", [])]
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
            for p in pair_data
        ]
        return cls(pairs=pairs, problematic_regions=regions)


@dataclass
class ChapterMarker:
    title: str
    start_ms: int
    end_ms: int


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
    author: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BookMeta:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class ExportConfig:
    intra_gap_ms: int = 300
    inter_gap_ms: int = 600
    format: str = "m4b"
    order: str = "l1-first"
    crossfade_ms: int = 30
    padding_ms: int = 75
    embed_cover: bool = True
    embed_chapters: bool = True
    warn_noise: bool = True
