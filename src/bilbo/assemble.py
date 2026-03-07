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
    preprocess_audio,
    slice_audio,
)
from .models import Alignment, AlignmentPair, ExportConfig

if TYPE_CHECKING:
    from .log import PipelineLog


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
) -> None:
    target_sr = 24000

    pp = log.parallel(["L1", "L2"], "Preprocessing", unit="s") if log else None
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(
            preprocess_audio, l1_audio_path, target_sr,
            on_progress=pp.callback("L1") if pp else None,
        )
        f2 = pool.submit(
            preprocess_audio, l2_audio_path, target_sr,
            on_progress=pp.callback("L2") if pp else None,
        )
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

        with AudioExporter(sr, channels, output_path, config.format) as exporter:
            for pi, pair in enumerate(pairs):
                chunk1 = apply_fade(_extract_chunk(pair, first_wav, sr, first_lang, config), sr)
                chunk2 = apply_fade(_extract_chunk(pair, second_wav, sr, second_lang, config), sr)

                if len(chunk1) > 0:
                    exporter.write(chunk1)
                if len(chunk1) > 0 and len(chunk2) > 0:
                    exporter.write(intra_gap)
                if len(chunk2) > 0:
                    exporter.write(chunk2)
                if pi < len(pairs) - 1:
                    exporter.write(inter_gap)

                if p:
                    p.update(pi + 1, len(pairs))

        if exporter.total_samples == 0:
            if log:
                log.warn("no audio content to assemble")
            return

        if p:
            p.finish(f"{exporter.duration / 60:.1f} minutes")

    finally:
        l1_wav.unlink(missing_ok=True)
        l2_wav.unlink(missing_ok=True)
