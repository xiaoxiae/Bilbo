from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Alignment, ChapterMarker

if TYPE_CHECKING:
    from .log import PipelineLog

COVER_JPEG_QUALITY = 90


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


def probe_metadata(audio_path: Path, log: PipelineLog | None = None) -> SourceMetadata:
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
        if log:
            log.warn(f"ffprobe failed for {audio_path.name}, metadata will be empty")
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
    """Merge two cover images with a diagonal split (L1 top-left, L2 bottom-right)."""
    from PIL import Image

    import numpy as np

    with Image.open(l1_cover) as img1, Image.open(l2_cover) as img2:
        # Resize both to the same dimensions
        target_w = min(img1.width, img2.width)
        target_h = min(img1.height, img2.height)
        if img1.size != (target_w, target_h):
            img1 = img1.resize((target_w, target_h), Image.LANCZOS)
        if img2.size != (target_w, target_h):
            img2 = img2.resize((target_w, target_h), Image.LANCZOS)

        arr1 = np.array(img1)
        arr2 = np.array(img2)

    ys = np.arange(target_h)
    cut_xs = (target_w * (1 - ys / target_h)).astype(int)
    xs = np.arange(target_w)
    mask = xs[np.newaxis, :] < cut_xs[:, np.newaxis]
    if arr2.ndim == 3:
        mask = mask[:, :, np.newaxis]
    arr2 = np.where(mask, arr1, arr2)
    Image.fromarray(arr2).save(str(output), "JPEG", quality=COVER_JPEG_QUALITY)


def _assign_chapters(
    midpoints: list[float],
    chapters: list[SourceChapter],
) -> list[int]:
    """Assign each midpoint to the chapter containing it."""
    result: list[int] = []
    for mid in midpoints:
        assigned = 0
        for ci, ch in enumerate(chapters):
            if ch.start <= mid <= ch.end:
                assigned = ci
                break
            if mid > ch.end and ci < len(chapters) - 1:
                continue
            assigned = ci
        result.append(assigned)
    return result


def _dedup_ordered(items: list[str]) -> list[str]:
    """Deduplicate a list while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def map_chapters_to_output(
    l1_chapters: list[SourceChapter],
    l2_chapters: list[SourceChapter],
    alignment: Alignment,
    pair_offsets_ms: list[tuple[int, int]],
    llm_merge: bool = False,
) -> list[ChapterMarker]:
    """Map source chapters to output timestamps using pair offsets.

    Algorithm:
    1. For each pair, compute the midpoint of its L1 and L2 segments
    2. Assign each pair to its L1 chapter (drives output chapter boundaries)
    3. Also assign each pair to its L2 chapter (for title building)
    4. Group by L1 chapter; title = "L1 title / unique L2 titles"
    """
    if not l1_chapters or not pair_offsets_ms:
        return []

    # Compute midpoints for L1 and L2
    l1_midpoints: list[float] = []
    l2_midpoints: list[float] = []
    for pair in alignment.pairs:
        if pair.l1:
            start = min(s.start for s in pair.l1)
            end = max(s.end for s in pair.l1)
            l1_midpoints.append((start + end) / 2)
        else:
            l1_midpoints.append(0.0)
        if pair.l2:
            start = min(s.start for s in pair.l2)
            end = max(s.end for s in pair.l2)
            l2_midpoints.append((start + end) / 2)
        else:
            l2_midpoints.append(0.0)

    # Assign each pair to L1 and L2 chapters independently
    pair_l1_ch = _assign_chapters(l1_midpoints, l1_chapters)
    pair_l2_ch = _assign_chapters(l2_midpoints, l2_chapters) if l2_chapters else []

    # Group pairs by L1 chapter and build markers + chapter_pairs for LLM
    markers: list[ChapterMarker] = []
    chapter_pairs: list[tuple[str, list[str]]] = []
    current_ch = -1
    ch_start_ms = 0
    ch_end_ms = 0
    l2_titles_in_group: list[str] = []

    def _finish_chapter() -> None:
        nonlocal ch_end_ms
        l1_title = l1_chapters[current_ch].title
        unique_l2 = _dedup_ordered(l2_titles_in_group)
        l2_part = ", ".join(unique_l2)
        title = f"{l1_title} / {l2_part}" if l2_part else l1_title
        markers.append(ChapterMarker(title=title, start_ms=ch_start_ms, end_ms=ch_end_ms))
        chapter_pairs.append((l1_title, unique_l2))

    for pi, ch_idx in enumerate(pair_l1_ch):
        if ch_idx != current_ch:
            if current_ch >= 0:
                _finish_chapter()

            current_ch = ch_idx
            ch_start_ms = pair_offsets_ms[pi][0]
            l2_titles_in_group = []

        ch_end_ms = pair_offsets_ms[pi][1]
        if pair_l2_ch and pi < len(pair_l2_ch):
            l2_titles_in_group.append(l2_chapters[pair_l2_ch[pi]].title)

    # Finish last chapter
    if current_ch >= 0:
        _finish_chapter()

    # Optionally refine chapter titles via LLM
    if llm_merge and markers and l2_chapters:
        from .llm import is_available, merge_chapter_titles
        if is_available():
            merged_titles = merge_chapter_titles(chapter_pairs)
            for i, title in enumerate(merged_titles):
                if i < len(markers):
                    markers[i] = ChapterMarker(
                        title=title,
                        start_ms=markers[i].start_ms,
                        end_ms=markers[i].end_ms,
                    )

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
