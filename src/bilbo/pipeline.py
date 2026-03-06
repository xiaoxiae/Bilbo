from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from .models import (
    Alignment,
    BookMeta,
    ExportConfig,
    Segment,
    SegmentedText,
    Word,
)
from .library import Library


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
) -> BookMeta:
    lib = library or Library()
    lib.init()

    slug = lib.make_slug(title)
    existing = lib.get(slug)
    if existing and not force:
        slug = existing.slug
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
    raw_l1_path = book_dir / "raw_segments_l1.json"
    raw_l2_path = book_dir / "raw_segments_l2.json"

    if force or not raw_l1_path.exists():
        click.echo("Stage 1: Transcribing L1...")
        from .transcribe import transcribe
        raw_l1 = transcribe(l1_audio, l1_lang, model_size, device)
        _save_raw_segments(raw_l1, raw_l1_path)
    else:
        click.echo("Stage 1: L1 transcription exists, skipping.")
        raw_l1 = _load_raw_segments(raw_l1_path)

    if force or not raw_l2_path.exists():
        click.echo("Stage 1: Transcribing L2...")
        from .transcribe import transcribe
        raw_l2 = transcribe(l2_audio, l2_lang, model_size, device)
        _save_raw_segments(raw_l2, raw_l2_path)
    else:
        click.echo("Stage 1: L2 transcription exists, skipping.")
        raw_l2 = _load_raw_segments(raw_l2_path)

    if 1 not in meta.stages_completed:
        meta.stages_completed.append(1)
    lib.add_or_update(meta)

    # Stage 2: Segmentation
    seg_l1_path = book_dir / "segments_l1.json"
    seg_l2_path = book_dir / "segments_l2.json"

    if force or not seg_l1_path.exists():
        click.echo("Stage 2: Segmenting L1...")
        from .segment import segment_text
        seg_l1 = segment_text(raw_l1, l1_lang)
        seg_l1.save(seg_l1_path)
    else:
        click.echo("Stage 2: L1 segmentation exists, skipping.")
        seg_l1 = SegmentedText.load(seg_l1_path)

    if force or not seg_l2_path.exists():
        click.echo("Stage 2: Segmenting L2...")
        from .segment import segment_text
        seg_l2 = segment_text(raw_l2, l2_lang)
        seg_l2.save(seg_l2_path)
    else:
        click.echo("Stage 2: L2 segmentation exists, skipping.")
        seg_l2 = SegmentedText.load(seg_l2_path)

    if 2 not in meta.stages_completed:
        meta.stages_completed.append(2)
    lib.add_or_update(meta)

    # Stage 3: Alignment
    align_path = book_dir / "alignment.json"

    if force or not align_path.exists():
        click.echo("Stage 3: Aligning...")
        from .align import align_texts
        alignment = align_texts(seg_l1, seg_l2, device=device)
        alignment.save(align_path)
    else:
        click.echo("Stage 3: Alignment exists, skipping.")
        alignment = Alignment.load(align_path)

    if 3 not in meta.stages_completed:
        meta.stages_completed.append(3)
    lib.add_or_update(meta)

    # Stage 4: Export
    if no_export:
        click.echo("Skipping export (--no-export).")
        return meta

    config = export_config or ExportConfig()
    click.echo("Stage 4: Assembling...")
    from .assemble import assemble

    output_name = f"interleaved.{config.format}"
    output_path = book_dir / "exports" / output_name
    assemble(alignment, l1_audio, l2_audio, config, output_path)

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
    lib = library or Library()
    meta = lib.get(slug)
    if meta is None:
        raise click.ClickException(f"Book '{slug}' not found in library.")

    book_dir = lib.book_dir(slug)
    align_path = book_dir / "alignment.json"

    if not align_path.exists():
        raise click.ClickException("Alignment not found. Run 'process' first.")

    alignment = Alignment.load(align_path)

    l1_audio = Path(meta.l1_audio)
    l2_audio = Path(meta.l2_audio)
    if not l1_audio.exists():
        raise click.ClickException(f"L1 audio not found: {l1_audio}")
    if not l2_audio.exists():
        raise click.ClickException(f"L2 audio not found: {l2_audio}")

    from .assemble import assemble

    output_name = f"interleaved.{config.format}"
    output_path = book_dir / "exports" / output_name

    click.echo(f"Exporting '{meta.title}'...")
    assemble(alignment, l1_audio, l2_audio, config, output_path)

    if output_name not in meta.exports:
        meta.exports.append(output_name)
    lib.add_or_update(meta)
