from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import click

from .library import Library
from .log import PipelineLog
from .models import (
    Alignment,
    BookMeta,
    ExportConfig,
    Segment,
    SegmentedText,
    Word,
)


_LOW_SCORE_THRESHOLD = 0.3


def _export_alignment_text(alignment: Alignment, output_path: Path) -> None:
    lines: list[str] = []
    for pair in alignment.pairs:
        l1_text = " ".join(s.text for s in pair.l1)
        l2_text = " ".join(s.text for s in pair.l2)
        l1_start = pair.l1[0].start
        l1_end = pair.l1[-1].end
        l2_start = pair.l2[0].start
        l2_end = pair.l2[-1].end
        marker = " !!!" if pair.score < _LOW_SCORE_THRESHOLD else ""
        lines.append(f"[{pair.score:.2f}]{marker}")
        lines.append(f"L1 ({l1_start:.2f}-{l1_end:.2f}): {l1_text}")
        lines.append(f"L2 ({l2_start:.2f}-{l2_end:.2f}): {l2_text}")
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


def run_pipeline(
    l1_audio: Path,
    l2_audio: Path,
    l1_lang: str,
    l2_lang: str,
    title: str,
    model_size: str = "large-v3-turbo",
    device: str = "auto",
    no_export: bool = False,
    export_config: ExportConfig | None = None,
    force: bool = False,
    library: Library | None = None,
    batch_size: int | None = None,
) -> BookMeta:
    log = PipelineLog()
    lib = library or Library()
    lib.init()

    existing_slug = lib.find_slug(title)
    if existing_slug:
        slug = existing_slug
        existing = lib.get(slug)
    else:
        slug = lib.make_slug(title)
        existing = None
    book_dir = lib.book_dir(slug)
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "exports").mkdir(exist_ok=True)

    meta = existing or BookMeta(
        slug=slug,
        title=title,
        l1_lang=l1_lang,
        l2_lang=l2_lang,
        l1_audio=str(l1_audio.resolve()),
        l2_audio=str(l2_audio.resolve()),
    )

    # Stage 1: Transcription
    log.stage(1, "Transcription")
    raw_l1_path = book_dir / "raw_segments_l1.json"
    raw_l2_path = book_dir / "raw_segments_l2.json"

    need_l1 = force or not raw_l1_path.exists()
    need_l2 = force or not raw_l2_path.exists()

    if need_l1 or need_l2:
        from .transcribe import transcribe, load_whisper_model

        log.info(f"Loading Whisper model ({model_size}) on {device}...")
        model = load_whisper_model(model_size, device)

        if need_l1 and need_l2:
            pp = log.parallel(["L1", "L2"], "Transcribing", unit="s")

            def _transcribe_and_save(audio_path, lang, out_path, label):
                segs = transcribe(
                    audio_path, lang, model=model,
                    batch_size=batch_size, on_progress=pp.callback(label),
                )
                _save_raw_segments(segs, out_path)
                return segs

            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_transcribe_and_save, l1_audio, l1_lang, raw_l1_path, "L1")
                f2 = pool.submit(_transcribe_and_save, l2_audio, l2_lang, raw_l2_path, "L2")
                raw_l1 = f1.result()
                raw_l2 = f2.result()
            pp.finish(f"L1: {len(raw_l1)} segments, L2: {len(raw_l2)} segments")
        elif need_l1:
            p = log.progress("Transcribing L1", unit="s")
            raw_l1 = transcribe(
                l1_audio, l1_lang, model=model,
                batch_size=batch_size, on_progress=p.update,
            )
            _save_raw_segments(raw_l1, raw_l1_path)
            p.finish(f"{len(raw_l1)} segments")
            raw_l2 = _load_raw_segments(raw_l2_path)
        else:
            raw_l1 = _load_raw_segments(raw_l1_path)
            p = log.progress("Transcribing L2", unit="s")
            raw_l2 = transcribe(
                l2_audio, l2_lang, model=model,
                batch_size=batch_size, on_progress=p.update,
            )
            _save_raw_segments(raw_l2, raw_l2_path)
            p.finish(f"{len(raw_l2)} segments")

        del model
    else:
        log.skip("cached")
        raw_l1 = _load_raw_segments(raw_l1_path)
        raw_l2 = _load_raw_segments(raw_l2_path)

    if 1 not in meta.stages_completed:
        meta.stages_completed.append(1)
    lib.add_or_update(meta)

    # Stage 2: Segmentation
    log.stage(2, "Segmentation")
    seg_l1_path = book_dir / "segments_l1.json"
    seg_l2_path = book_dir / "segments_l2.json"

    need_seg_l1 = force or not seg_l1_path.exists()
    need_seg_l2 = force or not seg_l2_path.exists()

    if not need_seg_l1 and not need_seg_l2:
        log.skip("cached")
        seg_l1 = SegmentedText.load(seg_l1_path)
        seg_l2 = SegmentedText.load(seg_l2_path)
    else:
        from .segment import segment_text

        def _segment_and_save(raw_segs, lang, out_path):
            result = segment_text(raw_segs, lang)
            result.save(out_path)
            return result

        if need_seg_l1 and need_seg_l2:
            log.info("Segmenting L1 + L2 in parallel...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_segment_and_save, raw_l1, l1_lang, seg_l1_path)
                f2 = pool.submit(_segment_and_save, raw_l2, l2_lang, seg_l2_path)
                seg_l1 = f1.result()
                seg_l2 = f2.result()
            log.done(f"L1: {len(seg_l1.sentences)} sentences, L2: {len(seg_l2.sentences)} sentences")
        elif need_seg_l1:
            log.info("Segmenting L1...")
            seg_l1 = _segment_and_save(raw_l1, l1_lang, seg_l1_path)
            log.done(f"L1: {len(seg_l1.sentences)} sentences")
            seg_l2 = SegmentedText.load(seg_l2_path)
        else:
            raw_l1 = _load_raw_segments(raw_l1_path)
            seg_l1 = SegmentedText.load(seg_l1_path)
            log.info("Segmenting L2...")
            seg_l2 = _segment_and_save(raw_l2, l2_lang, seg_l2_path)
            log.done(f"L2: {len(seg_l2.sentences)} sentences")

    if 2 not in meta.stages_completed:
        meta.stages_completed.append(2)
    lib.add_or_update(meta)

    # Stage 3: Alignment
    log.stage(3, "Alignment")
    align_path = book_dir / "alignment.json"

    if force or not align_path.exists():
        from .align import align_texts
        alignment = align_texts(seg_l1, seg_l2, device=device, log=log)
        alignment.save(align_path)
    else:
        log.skip("cached")
        alignment = Alignment.load(align_path)

    low_pairs = [p for p in alignment.pairs if p.score < _LOW_SCORE_THRESHOLD]
    if low_pairs:
        total_dur = sum(p.l1[-1].end - p.l1[0].start + p.l2[-1].end - p.l2[0].start for p in alignment.pairs)
        low_dur = sum(p.l1[-1].end - p.l1[0].start + p.l2[-1].end - p.l2[0].start for p in low_pairs)
        pct = low_dur / total_dur * 100 if total_dur > 0 else 0
        log.warn(f"{len(low_pairs)}/{len(alignment.pairs)} pairs misaligned ({pct:.1f}% of audio)")

    if 3 not in meta.stages_completed:
        meta.stages_completed.append(3)
    lib.add_or_update(meta)

    # Stage 4: Export
    if no_export:
        log.info("Skipping export (--no-export).")
        return meta

    config = export_config or ExportConfig()
    output_name = f"interleaved.{config.format}"
    output_path = book_dir / "exports" / output_name

    if config.format == "txt":
        log.stage(4, "Text export")
        _export_alignment_text(alignment, output_path)
        log.done(f"Written to {output_path}")
    else:
        log.stage(4, "Assembly")
        from .assemble import assemble
        assemble(alignment, l1_audio, l2_audio, config, output_path, log=log)

    if output_name not in meta.exports:
        meta.exports.append(output_name)
    if 4 not in meta.stages_completed:
        meta.stages_completed.append(4)
    lib.add_or_update(meta)

    return meta


def run_export(
    slug: str,
    config: ExportConfig,
    library: Library | None = None,
) -> None:
    log = PipelineLog()
    lib = library or Library()
    meta = lib.get(slug)
    if meta is None:
        raise click.ClickException(f"Book '{slug}' not found in library.")

    book_dir = lib.book_dir(slug)
    align_path = book_dir / "alignment.json"

    if not align_path.exists():
        raise click.ClickException("Alignment not found. Run 'process' first.")

    alignment = Alignment.load(align_path)

    output_name = f"interleaved.{config.format}"
    output_path = book_dir / "exports" / output_name

    if config.format == "txt":
        log.stage(4, "Text export")
        _export_alignment_text(alignment, output_path)
        log.done(f"Written to {output_path}")
    else:
        l1_audio = Path(meta.l1_audio)
        l2_audio = Path(meta.l2_audio)
        if not l1_audio.exists():
            raise click.ClickException(f"L1 audio not found: {l1_audio}")
        if not l2_audio.exists():
            raise click.ClickException(f"L2 audio not found: {l2_audio}")
        from .assemble import assemble
        log.stage(4, "Assembly")
        assemble(alignment, l1_audio, l2_audio, config, output_path, log=log)

    if output_name not in meta.exports:
        meta.exports.append(output_name)
    lib.add_or_update(meta)
