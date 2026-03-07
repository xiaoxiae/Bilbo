from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from .audio import (
    AudioExporter,
    apply_fade,
    generate_silence,
    generate_tone,
    post_process_metadata,
    preprocess_audio,
    slice_audio,
)
from .models import Alignment, AlignmentPair, ChapterMarker, ExportConfig

if TYPE_CHECKING:
    from .log import PipelineLog
    from .metadata import SourceMetadata


def _build_text_meta(
    l1_meta: SourceMetadata,
    l2_meta: SourceMetadata,
    config: ExportConfig,
    log: PipelineLog | None,  # noqa: ARG001
) -> dict[str, str]:
    """Build merged text metadata dict from L1/L2 source metadata."""
    pairs: dict[str, tuple[str, str]] = {}
    singles: dict[str, str] = {}
    for key, l1_val, l2_val in [
        ("title", l1_meta.title, l2_meta.title),
        ("artist", l1_meta.artist, l2_meta.artist),
        ("album", l1_meta.album, l2_meta.album),
        ("comment", l1_meta.comment, l2_meta.comment),
    ]:
        if l1_val and l2_val:
            pairs[key] = (l1_val, l2_val)
        elif l1_val:
            singles[key] = l1_val
        elif l2_val:
            singles[key] = l2_val

    if not pairs and not singles:
        return {}

    if pairs and config.llm_merge:
        from .llm import is_available, merge_metadata_text
        if is_available():
            merged = merge_metadata_text(pairs)
        else:
            merged = {k: " / ".join(v) for k, v in pairs.items()}
    else:
        merged = {k: " / ".join(v) for k, v in pairs.items()}

    return {**singles, **merged}


def _extract_chunk(
    pair: AlignmentPair,
    audio_path: Path,
    sr: int,
    lang: str,
    config: ExportConfig,
) -> np.ndarray:
    segs = pair.l1 if lang == "l1" else pair.l2

    if not segs:
        info = sf.info(str(audio_path))
        return np.zeros((0, info.channels), dtype=np.float32)

    start = min(s.start for s in segs)
    end = max(s.end for s in segs)
    return slice_audio(audio_path, sr, start, end, config.padding_ms)


def assemble(
    alignment: Alignment,
    l1_audio_path: Path,
    l2_audio_path: Path,
    config: ExportConfig,
    output_path: Path,
    log: PipelineLog | None = None,
    metadata: tuple[SourceMetadata, SourceMetadata] | None = None,
    cover_path: Path | None = None,
) -> None:
    target_sr = 24000

    pp = log.parallel(["L1", "L2"], "Preprocessing", unit="s") if log else None
    with ThreadPoolExecutor(max_workers=3) as pool:
        f1 = pool.submit(
            preprocess_audio, l1_audio_path, target_sr,
            on_progress=pp.callback("L1") if pp else None,
        )
        f2 = pool.submit(
            preprocess_audio, l2_audio_path, target_sr,
            on_progress=pp.callback("L2") if pp else None,
        )
        # Run LLM metadata merge in parallel with preprocessing
        llm_future = None
        if metadata and config.llm_merge:
            llm_future = pool.submit(_build_text_meta, metadata[0], metadata[1], config, None)
        l1_wav = f1.result()
        l2_wav = f2.result()

    try:
        l1_info = sf.info(str(l1_wav))
        sr = l1_info.samplerate
        channels = l1_info.channels

        if pp:
            l2_info = sf.info(str(l2_wav))
            l1_dur = l1_info.frames / sr
            l2_dur = l2_info.frames / sr
            pp.finish(f"Preprocessed (L1: {l1_dur:.0f}s, L2: {l2_dur:.0f}s)")

        pairs = alignment.pairs
        p = log.progress("Assembling") if log else None

        intra_gap = generate_silence(sr, config.intra_gap_ms, channels)
        inter_gap = generate_silence(sr, config.inter_gap_ms, channels)

        first_lang, second_lang = ("l1", "l2") if config.order == "l1-first" else ("l2", "l1")
        first_wav = l1_wav if first_lang == "l1" else l2_wav
        second_wav = l2_wav if first_lang == "l1" else l1_wav

        # Collect LLM-merged metadata (ran in parallel with preprocessing)
        text_meta: dict[str, str] | None = None
        if llm_future is not None:
            text_meta = llm_future.result()
            if log and text_meta:
                log.done("Metadata merged via LLM")
                for k, v in text_meta.items():
                    log.info(f"  {k}: {v}")
        elif metadata:
            l1_meta, l2_meta = metadata
            text_meta = _build_text_meta(l1_meta, l2_meta, config, None)

        pair_offsets_ms: list[tuple[int, int]] = []

        # Build warning tone data if enabled
        region_starts: set[int] = set()
        region_ends: set[int] = set()
        tone_gap: np.ndarray | None = None
        start_tone: np.ndarray | None = None
        end_tone: np.ndarray | None = None
        if config.warn_noise and alignment.problematic_regions:
            for rs, re in alignment.problematic_regions:
                region_starts.add(rs)
                region_ends.add(re)
            tone_gap = generate_silence(sr, 100, channels)
            start_tone = generate_tone(sr, 520, 200, channels, amplitude=0.3)
            end_tone = generate_tone(sr, 380, 200, channels, amplitude=0.3)

        with AudioExporter(sr, channels, output_path, config.format, metadata=text_meta) as exporter:
            for pi, pair in enumerate(pairs):
                start_ms = int(exporter.total_samples * 1000 / sr)

                if pi in region_starts:
                    assert start_tone is not None and tone_gap is not None
                    exporter.write(start_tone)
                    exporter.write(tone_gap)

                chunk1 = apply_fade(_extract_chunk(pair, first_wav, sr, first_lang, config), sr)
                chunk2 = apply_fade(_extract_chunk(pair, second_wav, sr, second_lang, config), sr)

                if len(chunk1) > 0:
                    exporter.write(chunk1)
                if len(chunk1) > 0 and len(chunk2) > 0:
                    exporter.write(intra_gap)
                if len(chunk2) > 0:
                    exporter.write(chunk2)

                if pi in region_ends:
                    assert end_tone is not None and tone_gap is not None
                    exporter.write(tone_gap)
                    exporter.write(end_tone)

                if pi < len(pairs) - 1:
                    exporter.write(inter_gap)

                end_ms = int(exporter.total_samples * 1000 / sr)
                pair_offsets_ms.append((start_ms, end_ms))

                if p:
                    p.update(pi + 1, len(pairs))

        if exporter.total_samples == 0:
            if log:
                log.warn("no audio content to assemble")
            return

        if p:
            p.finish(f"{exporter.duration / 60:.1f} minutes")

        # Post-process: embed cover art and chapters
        out_file = output_path.with_suffix(f".{config.format}")
        need_cover = config.embed_cover and cover_path and cover_path.exists()
        need_chapters = config.embed_chapters and metadata and (
            metadata[0].chapters or metadata[1].chapters
        )

        if need_cover or need_chapters:
            mapped_chapters: list[ChapterMarker] | None = None
            if need_chapters:
                from .metadata import map_chapters_to_output
                l1_meta, l2_meta = metadata  # type: ignore[misc]
                mapped_chapters = map_chapters_to_output(
                    l1_meta.chapters, l2_meta.chapters,
                    alignment, pair_offsets_ms, config.order,
                    llm_merge=config.llm_merge,
                )
                if log and mapped_chapters:
                    log.done(f"{len(mapped_chapters)} output chapters")
            post_process_metadata(
                out_file,
                cover_path=cover_path if need_cover else None,
                chapters=mapped_chapters,
            )

    finally:
        l1_wav.unlink(missing_ok=True)
        l2_wav.unlink(missing_ok=True)
