from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .library import Library
from .log import PipelineLog
from .metadata import (
    extract_cover_art,
    load_source_metadata,
    merge_covers,
    probe_metadata,
    save_source_metadata,
)
from .models import (
    Alignment,
    AlignmentPair,
    BookMeta,
    ExportConfig,
    Segment,
    SegmentedText,
    Word,
)


STAGE_NAMES = {1: "transcription", 2: "segmentation", 3: "alignment", 4: "export"}
STAGE_DIRS = {0: "0-input", 1: "1-transcribe", 2: "2-segment", 3: "3-align", 4: "4-export"}


def find_problematic_regions(
    pairs: list[AlignmentPair],
    window: int = 5,
    threshold: float = 0.35,
) -> list[tuple[int, int]]:
    """Find contiguous regions of poorly-aligned pairs using sliding window smoothing."""
    if not pairs:
        return []
    scores = [p.score for p in pairs]
    n = len(scores)
    half = window // 2
    import numpy as np
    cumsum = np.concatenate(([0.0], np.cumsum(scores)))
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(float((cumsum[hi] - cumsum[lo]) / (hi - lo)))

    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, s in enumerate(smoothed):
        if s < threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                regions.append((start, i - 1))
                start = None
    if start is not None:
        regions.append((start, n - 1))
    return regions


def _export_alignment_text(alignment: Alignment, output_path: Path) -> None:
    problematic_indices: set[int] = set()
    for start, end in alignment.problematic_regions:
        for i in range(start, end + 1):
            problematic_indices.add(i)

    lines: list[str] = []
    for i, pair in enumerate(alignment.pairs):
        marker = " !!!" if i in problematic_indices else ""
        lines.append(f"[{pair.score:.2f}]{marker}")
        if pair.l1:
            l1_text = " ".join(s.text for s in pair.l1)
            lines.append(f"L1 ({pair.l1[0].start:.2f}-{pair.l1[-1].end:.2f}): {l1_text}")
        else:
            lines.append("L1 (no audio)")
        if pair.l2:
            l2_text = " ".join(s.text for s in pair.l2)
            lines.append(f"L2 ({pair.l2[0].start:.2f}-{pair.l2[-1].end:.2f}): {l2_text}")
        else:
            lines.append("L2 (no audio)")
        lines.append("")
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    tmp.rename(output_path)


def _save_raw_segments(segments: list[Segment], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    data = [asdict(s) for s in segments]
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.rename(path)


def _load_raw_segments(path: Path) -> list[Segment]:
    data = json.loads(path.read_text())
    return [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=[Word(**w) for w in s.get("words", [])],
        )
        for s in data
    ]


def _prepare_metadata(
    l1_audio: Path,
    l2_audio: Path,
    book_dir: Path,
    force: bool,
    log: PipelineLog,
    l1_label: str = "L1",
    l2_label: str = "L2",
):
    """Extract, cache, and merge metadata from source audiobooks.

    Returns (metadata_tuple, cover_path) or (None, None) if nothing to embed.
    """
    meta_cache = book_dir / "source_metadata.json"

    if not force and meta_cache.exists():
        l1_meta, l2_meta = load_source_metadata(meta_cache)
        a = log.activity("Loading cached metadata...")
        a.done("Metadata loaded")
    else:
        a = log.activity("Extracting metadata...")
        l1_meta = probe_metadata(l1_audio, log=log)
        l2_meta = probe_metadata(l2_audio, log=log)
        save_source_metadata(l1_meta, l2_meta, meta_cache)
        a.done("Metadata extracted")

    # Log extracted metadata as dimmed details
    if l1_meta.title or l2_meta.title:
        l1_t = l1_meta.title or "?"
        l2_t = l2_meta.title or "?"
        log.detail(f"Titles: {l1_t} / {l2_t}")
    if l1_meta.artist or l2_meta.artist:
        l1_a = l1_meta.artist or "?"
        l2_a = l2_meta.artist or "?"
        log.detail(f"Artists: {l1_a} / {l2_a}")
    l1_ch = len(l1_meta.chapters)
    l2_ch = len(l2_meta.chapters)
    if l1_ch or l2_ch:
        log.detail(f"Chapters: {l1_label}={l1_ch}, {l2_label}={l2_ch}")
    if l1_meta.has_cover and l2_meta.has_cover:
        log.detail("Cover art: both sources")
    elif l1_meta.has_cover:
        log.detail(f"Cover art: {l1_label} only")
    elif l2_meta.has_cover:
        log.detail(f"Cover art: {l2_label} only")
    else:
        log.detail("Cover art: none")

    # Extract and merge covers
    cover_path: Path | None = None
    merged_cover = book_dir / "cover.jpg"
    if not force and merged_cover.exists():
        cover_path = merged_cover
    else:
        l1_cover = book_dir / "cover_l1.jpg"
        l2_cover = book_dir / "cover_l2.jpg"
        has_l1 = extract_cover_art(l1_audio, l1_cover) if l1_meta.has_cover else False
        has_l2 = extract_cover_art(l2_audio, l2_cover) if l2_meta.has_cover else False

        if has_l1 and has_l2:
            merge_covers(l1_cover, l2_cover, merged_cover)
            cover_path = merged_cover
        elif has_l1:
            l1_cover.rename(merged_cover)
            cover_path = merged_cover
        elif has_l2:
            l2_cover.rename(merged_cover)
            cover_path = merged_cover

        # Clean up individual covers
        l1_cover.unlink(missing_ok=True)
        l2_cover.unlink(missing_ok=True)

    metadata = (l1_meta, l2_meta)
    return metadata, cover_path


def run_pipeline(
    l1_audio: Path,
    l2_audio: Path,
    l1_lang: str | None = None,
    l2_lang: str | None = None,
    title: str | None = None,
    model_size: str = "large-v3",
    device: str = "auto",
    export_config: ExportConfig | None = None,
    library: Library | None = None,
    from_stage: int | None = None,
    to_stage: int | None = None,
) -> BookMeta:
    log = PipelineLog()

    # Resolve device: check CUDA availability, warn if requested but missing
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
    elif device == "cuda" and not cuda_available:
        log.warn("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    lib = library or Library()
    lib.init()

    force = [False] * 5  # index 0 unused; force[1..4] per stage
    if from_stage is not None:
        end = to_stage or 4
        for i in range(from_stage, end + 1):
            force[i] = True

    # Auto-generate title from filenames if not provided
    if title is None:
        stem1 = l1_audio.stem
        stem2 = l2_audio.stem
        title = stem1 if stem1 == stem2 else f"{stem1} + {stem2}"

    existing = lib.find_by_title(title)
    if existing:
        slug = existing.slug
    else:
        slug = lib.make_slug(title)
    book_dir = lib.book_dir(slug)
    book_dir.mkdir(parents=True, exist_ok=True)

    # Copy input audio into the book directory so we don't depend on
    # the original files staying in place.
    log.stage(0, "Input")
    input_dir = book_dir / STAGE_DIRS[0]
    input_dir.mkdir(exist_ok=True)
    l1_copy = input_dir / f"l1{l1_audio.suffix}"
    l2_copy = input_dir / f"l2{l2_audio.suffix}"
    need_copy = not l1_copy.exists() or not l2_copy.exists()
    if need_copy:
        import shutil
        a = log.activity("Copying input audio...")
        if not l1_copy.exists():
            shutil.copy2(l1_audio, l1_copy)
        if not l2_copy.exists():
            shutil.copy2(l2_audio, l2_copy)
        a.done("Input audio copied")
    else:
        log.skip("cached")
    l1_audio = l1_copy
    l2_audio = l2_copy

    meta = existing or BookMeta(
        slug=slug,
        title=title,
        l1_lang=l1_lang or "",
        l2_lang=l2_lang or "",
        l1_audio=str(l1_copy),
        l2_audio=str(l2_copy),
    )
    meta.l1_audio = str(l1_copy)
    meta.l2_audio = str(l2_copy)

    l1_label = l1_lang.upper() if l1_lang else "L1"
    l2_label = l2_lang.upper() if l2_lang else "L2"

    # Check prerequisites when starting from a later stage
    if from_stage is not None and from_stage > 1 and existing:
        for s in range(1, from_stage):
            if s not in meta.stages_completed:
                raise ValueError(
                    f"Stage {s} ({STAGE_NAMES[s]}) not completed. "
                    f"Run 'bilbo process --title \"{title}\" --from {s} --to {s}' first."
                )

    # Stage 1: Transcription
    log.stage(1, "Transcription")
    transcribe_dir = book_dir / STAGE_DIRS[1]
    transcribe_dir.mkdir(exist_ok=True)
    raw_l1_path = transcribe_dir / "raw_segments_l1.json"
    raw_l2_path = transcribe_dir / "raw_segments_l2.json"

    need_l1 = force[1] or not raw_l1_path.exists()
    need_l2 = force[1] or not raw_l2_path.exists()

    if need_l1 or need_l2:
        from .transcribe import transcribe, load_whisper_model

        a = log.activity("Loading Whisper model...", detail=f"({model_size}, {device})")
        model = load_whisper_model(model_size, device)
        a.done("Model loaded")

        if need_l1 and need_l2:
            p = log.progress(f"Transcribing {l1_label}", unit="s")
            raw_l1, det_l1 = transcribe(
                l1_audio, l1_lang, model=model,
                on_progress=p.update,
            )
            _save_raw_segments(raw_l1, raw_l1_path)
            p.finish(f"{len(raw_l1)} segments")

            p = log.progress(f"Transcribing {l2_label}", unit="s")
            raw_l2, det_l2 = transcribe(
                l2_audio, l2_lang, model=model,
                on_progress=p.update,
            )
            _save_raw_segments(raw_l2, raw_l2_path)
            p.finish(f"{len(raw_l2)} segments")
        elif need_l1:
            p = log.progress(f"Transcribing {l1_label}", unit="s")
            raw_l1, det_l1 = transcribe(
                l1_audio, l1_lang, model=model,
                on_progress=p.update,
            )
            _save_raw_segments(raw_l1, raw_l1_path)
            p.finish(f"{len(raw_l1)} segments")
            raw_l2 = _load_raw_segments(raw_l2_path)
            det_l2 = l2_lang
        else:
            raw_l1 = _load_raw_segments(raw_l1_path)
            det_l1 = l1_lang
            p = log.progress(f"Transcribing {l2_label}", unit="s")
            raw_l2, det_l2 = transcribe(
                l2_audio, l2_lang, model=model,
                on_progress=p.update,
            )
            _save_raw_segments(raw_l2, raw_l2_path)
            p.finish(f"{len(raw_l2)} segments")

        del model

        # Update langs from detection
        if not l1_lang:
            l1_lang = det_l1
            log.info(f"Detected L1 language: {l1_lang}")
        if not l2_lang:
            l2_lang = det_l2
            log.info(f"Detected L2 language: {l2_lang}")
        l1_label = l1_lang.upper()
        l2_label = l2_lang.upper()
        meta.l1_lang = l1_lang
        meta.l2_lang = l2_lang
    else:
        if not l1_lang or not l2_lang:
            raise ValueError(
                "Cannot auto-detect languages from cached transcription. "
                "Provide --l1/--l2 explicitly."
            )
        log.skip("cached")
        raw_l1 = _load_raw_segments(raw_l1_path)
        raw_l2 = _load_raw_segments(raw_l2_path)

    if 1 not in meta.stages_completed:
        meta.stages_completed.append(1)
    lib.add_or_update(meta)

    if to_stage is not None and to_stage < 2:
        log.info(f"Stopping after stage {to_stage}.")
        log.summary()
        return meta

    # Stage 2: Segmentation
    log.stage(2, "Segmentation")
    segment_dir = book_dir / STAGE_DIRS[2]
    segment_dir.mkdir(exist_ok=True)
    seg_l1_path = segment_dir / "segments_l1.json"
    seg_l2_path = segment_dir / "segments_l2.json"

    need_seg_l1 = force[2] or not seg_l1_path.exists()
    need_seg_l2 = force[2] or not seg_l2_path.exists()

    if not need_seg_l1 and not need_seg_l2:
        log.skip("cached")
        seg_l1 = SegmentedText.load(seg_l1_path)
        seg_l2 = SegmentedText.load(seg_l2_path)
    else:
        from .segment import refine_timestamps, segment_text

        def _segment_and_save(raw_segs, lang, audio, out_path):
            result = segment_text(raw_segs, lang)
            result, refine_stats = refine_timestamps(result, audio)
            result.save(out_path)
            return result, refine_stats

        def _log_refine_stats(label, stats):
            parts = []
            if stats["extended"]:
                parts.append(f"{stats['extended']} extended (avg {stats['avg_extend_ms']:.0f}ms)")
            if stats["contracted"]:
                parts.append(f"{stats['contracted']} contracted (avg {stats['avg_contract_ms']:.0f}ms)")
            if parts:
                log.detail(f"{label}: refined {stats['adjusted']}/{stats['total']} endpoints — {', '.join(parts)}")

        if need_seg_l1 and need_seg_l2:
            act = log.activity("Segmenting...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_segment_and_save, raw_l1, l1_lang, l1_audio, seg_l1_path)
                f2 = pool.submit(_segment_and_save, raw_l2, l2_lang, l2_audio, seg_l2_path)
                seg_l1, stats_l1 = f1.result()
                seg_l2, stats_l2 = f2.result()
            act.done(f"{l1_label}: {len(seg_l1.sentences)} sentences, {l2_label}: {len(seg_l2.sentences)} sentences")
            _log_refine_stats(l1_label, stats_l1)
            _log_refine_stats(l2_label, stats_l2)
        elif need_seg_l1:
            act = log.activity(f"Segmenting {l1_label}...")
            seg_l1, stats_l1 = _segment_and_save(raw_l1, l1_lang, l1_audio, seg_l1_path)
            act.done(f"{l1_label}: {len(seg_l1.sentences)} sentences")
            _log_refine_stats(l1_label, stats_l1)
            seg_l2 = SegmentedText.load(seg_l2_path)
        else:
            seg_l1 = SegmentedText.load(seg_l1_path)
            act = log.activity(f"Segmenting {l2_label}...")
            seg_l2, stats_l2 = _segment_and_save(raw_l2, l2_lang, l2_audio, seg_l2_path)
            act.done(f"{l2_label}: {len(seg_l2.sentences)} sentences")
            _log_refine_stats(l2_label, stats_l2)

    if 2 not in meta.stages_completed:
        meta.stages_completed.append(2)
    lib.add_or_update(meta)

    if to_stage is not None and to_stage < 3:
        log.info(f"Stopping after stage {to_stage}.")
        log.summary()
        return meta

    # Stage 3: Alignment
    log.stage(3, "Alignment")
    align_dir = book_dir / STAGE_DIRS[3]
    align_dir.mkdir(exist_ok=True)
    align_path = align_dir / "alignment.json"

    if force[3] or not align_path.exists():
        from .align import align_texts
        alignment = align_texts(seg_l1, seg_l2, device=device, log=log, book_dir=align_dir)
        alignment.problematic_regions = find_problematic_regions(alignment.pairs)
        alignment.save(align_path)
    else:
        log.skip("cached")
        alignment = Alignment.load(align_path)
        if not alignment.problematic_regions:
            alignment.problematic_regions = find_problematic_regions(alignment.pairs)

    problematic_indices: set[int] = set()
    for start, end in alignment.problematic_regions:
        for i in range(start, end + 1):
            problematic_indices.add(i)
    if problematic_indices:
        low_pairs = [alignment.pairs[i] for i in problematic_indices]
        def _pair_dur(p: AlignmentPair) -> float:
            d = 0.0
            if p.l1:
                d += p.l1[-1].end - p.l1[0].start
            if p.l2:
                d += p.l2[-1].end - p.l2[0].start
            return d
        total_dur = sum(_pair_dur(p) for p in alignment.pairs)
        low_dur = sum(_pair_dur(p) for p in low_pairs)
        pct = low_dur / total_dur * 100 if total_dur > 0 else 0
        log.warn(f"{len(low_pairs)}/{len(alignment.pairs)} pairs misaligned ({pct:.1f}% of audio)")

    if 3 not in meta.stages_completed:
        meta.stages_completed.append(3)
    lib.add_or_update(meta)

    # Stage 4: Export
    if to_stage is not None and to_stage < 4:
        log.info(f"Stopping after stage {to_stage}.")
        log.summary()
        return meta

    config = export_config or ExportConfig()
    export_dir = book_dir / STAGE_DIRS[4]
    export_dir.mkdir(exist_ok=True)
    output_name = f"interleaved.{config.format}"
    output_path = export_dir / output_name

    # Always generate text alignment alongside audio export
    txt_path = export_dir / "interleaved.txt"
    _export_alignment_text(alignment, txt_path)
    if "interleaved.txt" not in meta.exports:
        meta.exports.append("interleaved.txt")

    if force[4] or not output_path.exists():
        log.stage(4, "Assembly")
        from .assemble import assemble

        source_metadata, cover = _prepare_metadata(
            l1_audio, l2_audio, book_dir, force[4], log,
            l1_label=l1_label, l2_label=l2_label,
        )

        # Update author from extracted metadata
        if source_metadata:
            l1_m, l2_m = source_metadata
            if not meta.author:
                meta.author = l1_m.artist or l2_m.artist

        assemble(
            alignment, l1_audio, l2_audio, config, output_path,
            log=log, metadata=source_metadata, cover_path=cover,
            lang_labels=(l1_label, l2_label),
        )
    else:
        log.stage(4, "Assembly")
        log.skip("cached")

    if output_name not in meta.exports:
        meta.exports.append(output_name)
    if 4 not in meta.stages_completed:
        meta.stages_completed.append(4)
    lib.add_or_update(meta)

    log.summary()
    return meta
