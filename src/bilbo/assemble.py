from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from .audio import (
    crossfade,
    export_audio,
    generate_silence,
    load_audio,
    normalize_lufs,
    resample,
    slice_audio,
)
from .models import Alignment, AlignmentPair, ExportConfig, SegmentedText


def _extract_chunk(
    pairs: list[AlignmentPair],
    audio_data: np.ndarray,
    sr: int,
    lang: str,
    config: ExportConfig,
) -> np.ndarray:
    segs = []
    for p in pairs:
        segs.extend(p.l1 if lang == "l1" else p.l2)

    if not segs:
        return np.zeros((0, audio_data.shape[1] if audio_data.ndim > 1 else 1))

    # Take one continuous slice from first segment start to last segment end
    start = min(s.start for s in segs)
    end = max(s.end for s in segs)
    return slice_audio(audio_data, sr, start, end, config.padding_ms)


def assemble(
    alignment: Alignment,
    l1_audio_path: Path,
    l2_audio_path: Path,
    config: ExportConfig,
    output_path: Path,
) -> None:
    click.echo(f"  Loading audio files...")
    l1_data, l1_sr = load_audio(l1_audio_path)
    l2_data, l2_sr = load_audio(l2_audio_path)

    # Resample to common sample rate
    sr = max(l1_sr, l2_sr)
    if l1_sr != sr:
        click.echo(f"  Resampling L1 from {l1_sr}Hz to {sr}Hz...")
        l1_data = resample(l1_data, l1_sr, sr)
    if l2_sr != sr:
        click.echo(f"  Resampling L2 from {l2_sr}Hz to {sr}Hz...")
        l2_data = resample(l2_data, l2_sr, sr)

    channels = max(
        l1_data.shape[1] if l1_data.ndim > 1 else 1,
        l2_data.shape[1] if l2_data.ndim > 1 else 1,
    )

    click.echo(f"  Normalizing audio levels...")
    l1_data = normalize_lufs(l1_data, sr)
    l2_data = normalize_lufs(l2_data, sr)

    pairs = alignment.pairs
    click.echo(f"  Assembling {len(pairs)} pairs...")

    intra_gap = generate_silence(sr, config.intra_gap_ms, channels)
    inter_gap = generate_silence(sr, config.inter_gap_ms, channels)

    first_lang, second_lang = ("l1", "l2") if config.order == "l1-first" else ("l2", "l1")
    first_data = l1_data if first_lang == "l1" else l2_data
    second_data = l2_data if first_lang == "l1" else l1_data

    all_parts = []
    for pi, pair in enumerate(pairs):
        chunk1 = _extract_chunk([pair], first_data, sr, first_lang, config)
        chunk2 = _extract_chunk([pair], second_data, sr, second_lang, config)

        if len(chunk1) > 0:
            all_parts.append(chunk1)
        if len(chunk1) > 0 and len(chunk2) > 0:
            all_parts.append(intra_gap)
        if len(chunk2) > 0:
            all_parts.append(chunk2)
        if pi < len(pairs) - 1:
            all_parts.append(inter_gap)

        if (pi + 1) % 50 == 0 or pi == len(pairs) - 1:
            click.echo(f"    {pi + 1}/{len(pairs)} pairs done")

    if not all_parts:
        click.echo("  Warning: no audio content to assemble")
        return

    click.echo(f"  Concatenating...")
    result = all_parts[0]
    for p in all_parts[1:]:
        result = crossfade(result, p, config.crossfade_ms)

    click.echo(f"  Exporting to {output_path.name}...")
    export_audio(result, sr, output_path, config.format)
    duration = len(result) / sr
    click.echo(f"  Done! Output: {duration / 60:.1f} minutes")
