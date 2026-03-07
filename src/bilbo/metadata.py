from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .models import Alignment, ChapterMarker


@dataclass
class SourceChapter:
    title: str
    start: float
    end: float


@dataclass
class SourceMetadata:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    comment: str | None = None
    chapters: list[SourceChapter] = field(default_factory=list)
    has_cover: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SourceMetadata:
        chapters = [SourceChapter(**c) for c in d.get("chapters", [])]
        return cls(
            title=d.get("title"),
            artist=d.get("artist"),
            album=d.get("album"),
            comment=d.get("comment"),
            chapters=chapters,
            has_cover=d.get("has_cover", False),
        )


def probe_metadata(audio_path: Path) -> SourceMetadata:
    """Extract metadata, chapters, and cover art presence via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_chapters", "-show_streams",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return SourceMetadata()

    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    tags = fmt.get("tags", {})
    # Tags may have varying case
    tags_lower = {k.lower(): v for k, v in tags.items()}

    chapters = []
    for ch in data.get("chapters", []):
        ch_tags = ch.get("tags", {})
        ch_tags_lower = {k.lower(): v for k, v in ch_tags.items()}
        chapters.append(SourceChapter(
            title=ch_tags_lower.get("title", f"Chapter {len(chapters) + 1}"),
            start=float(ch.get("start_time", 0)),
            end=float(ch.get("end_time", 0)),
        ))

    has_cover = any(
        s.get("codec_type") == "video"
        or s.get("disposition", {}).get("attached_pic", 0) == 1
        for s in data.get("streams", [])
    )

    return SourceMetadata(
        title=tags_lower.get("title"),
        artist=tags_lower.get("artist") or tags_lower.get("album_artist"),
        album=tags_lower.get("album"),
        comment=tags_lower.get("comment") or tags_lower.get("description"),
        chapters=chapters,
        has_cover=has_cover,
    )


def extract_cover_art(audio_path: Path, output_path: Path) -> bool:
    """Extract cover art from audio file. Returns True if cover was found."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-an", "-vcodec", "copy",
            str(output_path),
        ],
        capture_output=True,
    )
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return True
    output_path.unlink(missing_ok=True)
    return False


def merge_covers(l1_cover: Path, l2_cover: Path, output: Path) -> None:
    """Merge two cover images side-by-side, resizing to same height."""
    from PIL import Image

    img1 = Image.open(l1_cover)
    img2 = Image.open(l2_cover)

    # Resize to same height (use the smaller one)
    target_h = min(img1.height, img2.height)
    if img1.height != target_h:
        w = int(img1.width * target_h / img1.height)
        img1 = img1.resize((w, target_h), Image.LANCZOS)
    if img2.height != target_h:
        w = int(img2.width * target_h / img2.height)
        img2 = img2.resize((w, target_h), Image.LANCZOS)

    merged = Image.new("RGB", (img1.width + img2.width, target_h))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))
    merged.save(str(output), "JPEG", quality=90)


def map_chapters_to_output(
    l1_chapters: list[SourceChapter],
    l2_chapters: list[SourceChapter],
    alignment: Alignment,
    pair_offsets_ms: list[tuple[int, int]],
    lang_order: str = "l1-first",  # noqa: ARG001
) -> list[ChapterMarker]:
    """Map source chapters to output timestamps using pair offsets.

    Algorithm:
    1. For each alignment pair, compute the midpoint of its L1 segments' timestamps
    2. Assign each pair to the L1 source chapter containing that midpoint
    3. The output chapter starts at the output offset of the first pair in the group
    4. Title = "L1 chapter title / L2 chapter title" (matched by index)
    """
    if not l1_chapters or not pair_offsets_ms:
        return []

    # Compute midpoint of L1 segments for each pair
    pair_midpoints: list[float] = []
    for pair in alignment.pairs:
        if pair.l1:
            start = min(s.start for s in pair.l1)
            end = max(s.end for s in pair.l1)
            pair_midpoints.append((start + end) / 2)
        else:
            pair_midpoints.append(0.0)

    # Assign each pair to a chapter
    pair_chapter_idx: list[int] = []
    for mid in pair_midpoints:
        assigned = 0
        for ci, ch in enumerate(l1_chapters):
            if ch.start <= mid <= ch.end:
                assigned = ci
                break
            # If midpoint is past this chapter, try next
            if mid > ch.end and ci < len(l1_chapters) - 1:
                continue
            # Default to last chapter if past all
            assigned = ci
        pair_chapter_idx.append(assigned)

    # Group pairs by chapter and build markers
    markers: list[ChapterMarker] = []
    current_ch = -1
    ch_start_ms = 0
    ch_end_ms = 0

    for pi, ch_idx in enumerate(pair_chapter_idx):
        if ch_idx != current_ch:
            # Finish previous chapter
            if current_ch >= 0:
                l1_title = l1_chapters[current_ch].title if current_ch < len(l1_chapters) else ""
                l2_title = l2_chapters[current_ch].title if current_ch < len(l2_chapters) else ""
                title = f"{l1_title} / {l2_title}" if l2_title else l1_title
                markers.append(ChapterMarker(title=title, start_ms=ch_start_ms, end_ms=ch_end_ms))

            current_ch = ch_idx
            ch_start_ms = pair_offsets_ms[pi][0]

        ch_end_ms = pair_offsets_ms[pi][1]

    # Finish last chapter
    if current_ch >= 0:
        l1_title = l1_chapters[current_ch].title if current_ch < len(l1_chapters) else ""
        l2_title = l2_chapters[current_ch].title if current_ch < len(l2_chapters) else ""
        title = f"{l1_title} / {l2_title}" if l2_title else l1_title
        markers.append(ChapterMarker(title=title, start_ms=ch_start_ms, end_ms=ch_end_ms))

    return markers


def save_source_metadata(
    l1_meta: SourceMetadata,
    l2_meta: SourceMetadata,
    path: Path,
) -> None:
    """Cache extracted metadata to JSON."""
    data = {"l1": l1_meta.to_dict(), "l2": l2_meta.to_dict()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.rename(path)


def load_source_metadata(path: Path) -> tuple[SourceMetadata, SourceMetadata]:
    """Load cached metadata from JSON."""
    data = json.loads(path.read_text())
    return SourceMetadata.from_dict(data["l1"]), SourceMetadata.from_dict(data["l2"])
