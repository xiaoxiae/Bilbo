from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .models import Alignment, ChapterMarker

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
    merged.save(str(output), "JPEG", quality=COVER_JPEG_QUALITY)


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


def map_chapters_to_output(
    l1_chapters: list[SourceChapter],
    l2_chapters: list[SourceChapter],
    alignment: Alignment,
    pair_offsets_ms: list[tuple[int, int]],
    lang_order: str = "l1-first",  # noqa: ARG001
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

    # Group pairs by L1 chapter and build markers
    markers: list[ChapterMarker] = []
    current_ch = -1
    ch_start_ms = 0
    ch_end_ms = 0
    l2_titles_in_group: list[str] = []

    for pi, ch_idx in enumerate(pair_l1_ch):
        if ch_idx != current_ch:
            # Finish previous chapter
            if current_ch >= 0:
                l1_title = l1_chapters[current_ch].title
                # Deduplicate L2 titles while preserving order
                seen: set[str] = set()
                unique_l2: list[str] = []
                for t in l2_titles_in_group:
                    if t not in seen:
                        seen.add(t)
                        unique_l2.append(t)
                l2_part = ", ".join(unique_l2)
                title = f"{l1_title} / {l2_part}" if l2_part else l1_title
                markers.append(ChapterMarker(title=title, start_ms=ch_start_ms, end_ms=ch_end_ms))

            current_ch = ch_idx
            ch_start_ms = pair_offsets_ms[pi][0]
            l2_titles_in_group = []

        ch_end_ms = pair_offsets_ms[pi][1]
        if pair_l2_ch and pi < len(pair_l2_ch):
            l2_titles_in_group.append(l2_chapters[pair_l2_ch[pi]].title)

    # Finish last chapter
    if current_ch >= 0:
        l1_title = l1_chapters[current_ch].title
        seen = set()
        unique_l2 = []
        for t in l2_titles_in_group:
            if t not in seen:
                seen.add(t)
                unique_l2.append(t)
        l2_part = ", ".join(unique_l2)
        title = f"{l1_title} / {l2_part}" if l2_part else l1_title
        markers.append(ChapterMarker(title=title, start_ms=ch_start_ms, end_ms=ch_end_ms))

    # Optionally refine chapter titles via LLM
    if llm_merge and markers and l2_chapters:
        from .llm import is_available, merge_chapter_titles
        if is_available():
            # Rebuild (l1_title, [l2_titles]) pairs for LLM merging
            chapter_pairs: list[tuple[str, list[str]]] = []
            marker_idx = 0
            current_ch = -1
            l2_titles_group: list[str] = []
            l1_title_for_ch = ""
            for pi, ch_idx in enumerate(pair_l1_ch):
                if ch_idx != current_ch:
                    if current_ch >= 0:
                        seen_set: set[str] = set()
                        deduped: list[str] = []
                        for t in l2_titles_group:
                            if t not in seen_set:
                                seen_set.add(t)
                                deduped.append(t)
                        chapter_pairs.append((l1_title_for_ch, deduped))
                        marker_idx += 1
                    current_ch = ch_idx
                    l1_title_for_ch = l1_chapters[ch_idx].title
                    l2_titles_group = []
                if pair_l2_ch and pi < len(pair_l2_ch):
                    l2_titles_group.append(l2_chapters[pair_l2_ch[pi]].title)
            if current_ch >= 0:
                seen_set = set()
                deduped = []
                for t in l2_titles_group:
                    if t not in seen_set:
                        seen_set.add(t)
                        deduped.append(t)
                chapter_pairs.append((l1_title_for_ch, deduped))

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
