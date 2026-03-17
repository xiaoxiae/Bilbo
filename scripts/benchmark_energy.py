"""Visualize RMS energy around segment boundaries at multiple window sizes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bilbo.library import Library
from bilbo.models import Segment, Word
from bilbo.segment import _decode_to_wav, segment_text


def compute_rms(samples: np.ndarray, sr: int, win_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMS energy in non-overlapping windows.

    Returns (times, energies) where times is the center of each window.
    """
    win_samples = max(1, int(sr * win_ms / 1000))
    n_windows = len(samples) // win_samples
    if n_windows == 0:
        return np.array([]), np.array([])
    # Trim to exact multiple
    trimmed = samples[: n_windows * win_samples].reshape(n_windows, win_samples)
    energies = np.sqrt(np.mean(trimmed ** 2, axis=1))
    centers = (np.arange(n_windows) + 0.5) * win_samples / sr
    return centers, energies


_VAD_PADDING = 0.05  # 50 ms guard (mirrors segment.py)


def compute_vad_refined_bounds(
    sentences: list[Segment], speech_regions: list[dict]
) -> list[tuple[float, float]]:
    """Apply ownership-based VAD snapping to compute (new_start, new_end) per segment."""
    bounds = []
    for i, seg in enumerate(sentences):
        prev_end = sentences[i - 1].end if i > 0 else 0.0
        next_start = sentences[i + 1].start if i < len(sentences) - 1 else float("inf")

        matched = []
        for region in speech_regions:
            r_start, r_end = region["start"], region["end"]
            dur = r_end - r_start
            if dur <= 0:
                continue
            overlap = max(0.0, min(r_end, seg.end) - max(r_start, seg.start))
            if overlap / dur > 0.5:
                matched.append(region)

        if not matched:
            bounds.append((seg.start, seg.end))
            continue

        new_start = max(min(r["start"] for r in matched) - _VAD_PADDING, prev_end)
        new_end = min(max(r["end"] for r in matched) + _VAD_PADDING, next_start)
        bounds.append((new_start, new_end))
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy profile around segment boundaries")
    parser.add_argument("book", help="Book title or numeric ID")
    parser.add_argument("--lang", choices=["l1", "l2"], default="l1")
    parser.add_argument("--start", type=float, default=60.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=120.0, help="End time in seconds")
    parser.add_argument(
        "--windows",
        default="30,100,300",
        help="Comma-separated window sizes in ms (default: 30,100,300)",
    )
    args = parser.parse_args()

    win_sizes = [float(w) for w in args.windows.split(",")]

    lib = Library()
    meta = lib.find(args.book)
    if meta is None:
        print(f"Book not found: {args.book}", file=sys.stderr)
        sys.exit(1)

    slug = meta.slug
    book_dir = lib.book_dir(slug)

    # Load raw segments and re-segment (without VAD) to get original ends
    lang_code = meta.l1_lang if args.lang == "l1" else meta.l2_lang
    raw_path = book_dir / "1-transcribe" / f"raw_segments_{args.lang}.json"
    if not raw_path.exists():
        print(f"Raw segments not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    import json
    raw_data = json.loads(raw_path.read_text())
    raw_segs = [
        Segment(
            start=s["start"], end=s["end"], text=s["text"],
            words=[Word(**w) for w in s.get("words", [])],
        )
        for s in raw_data
    ]
    original = segment_text(raw_segs, lang_code)
    print(f"Loaded {len(original.sentences)} sentences")

    # Decode audio
    audio_path = Path(meta.l1_audio if args.lang == "l1" else meta.l2_audio)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Decoding {audio_path} ({args.start}–{args.end}s)...")
    samples, sr = _decode_to_wav(audio_path)

    # Crop to range before VAD — no need to process the full file
    start_sample = int(args.start * sr)
    end_sample = int(args.end * sr)
    cropped = samples[start_sample:end_sample]

    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    print("Running Silero VAD on range...")
    model = load_silero_vad()
    cropped_tensor = torch.from_numpy(cropped)
    # Timestamps are relative to cropped start; shift by args.start for absolute coords
    speech_regions_rel = get_speech_timestamps(cropped_tensor, model, sampling_rate=sr, return_seconds=True)
    speech_regions = [{"start": r["start"] + args.start, "end": r["end"] + args.start} for r in speech_regions_rel]
    print(f"VAD found {len(speech_regions)} speech regions")

    # Compute per-chunk VAD probabilities (silero uses 512-sample chunks at 16kHz)
    chunk_size = 512
    model.reset_states()
    vad_probs: list[float] = []
    vad_times: list[float] = []
    for i in range(0, len(cropped_tensor), chunk_size):
        chunk = cropped_tensor[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
        vad_probs.append(model(chunk, sr).item())
        vad_times.append(args.start + (i + chunk_size / 2) / sr)
    vt = np.array(vad_times)
    vp = np.array(vad_probs)

    # Compute VAD-refined bounds for segments in range
    vad_bounds = compute_vad_refined_bounds(original.sentences, speech_regions)

    # Collect boundaries in range
    original_ends = [s.end for s in original.sentences if args.start <= s.end <= args.end]
    original_starts = [s.start for s in original.sentences if args.start <= s.start <= args.end]
    vad_ends_in_range = [end for _, end in vad_bounds if args.start <= end <= args.end]
    vad_starts_in_range = [start for start, _ in vad_bounds if args.start <= start <= args.end]

    # Plot
    fig, axes = plt.subplots(len(win_sizes), 1, figsize=(14, 3 * len(win_sizes)), sharex=True)
    if len(win_sizes) == 1:
        axes = [axes]

    for ax, win_ms in zip(axes, win_sizes):
        # VAD probability as background on a twin axis
        ax_vad = ax.twinx()
        ax_vad.fill_between(vt, 0, vp, alpha=0.25, color="limegreen", label="VAD prob")
        ax_vad.set_ylim(0, 1)
        ax_vad.tick_params(axis="y", labelcolor="green", labelsize=7)
        ax_vad.set_ylabel("VAD prob", fontsize=7, color="green")
        # Keep energy plot drawn on top of VAD fill
        ax.set_zorder(ax_vad.get_zorder() + 1)
        ax.patch.set_visible(False)

        times, energies = compute_rms(cropped, sr, win_ms)
        times = times + args.start
        ax.plot(times, energies, linewidth=0.5, color="black")
        ax.fill_between(times, energies, alpha=0.3, color="gray")

        for t in original_ends:
            ax.axvline(t, color="orange", linewidth=0.8, alpha=0.6, label="original .end")
        for t in original_starts:
            ax.axvline(t, color="blue", linewidth=0.8, linestyle="--", alpha=0.7, label="original .start")
        for t in vad_ends_in_range:
            ax.axvline(t, color="green", linewidth=1.0, alpha=0.8, label="VAD-refined .end")
        for t in vad_starts_in_range:
            ax.axvline(t, color="purple", linewidth=1.0, linestyle="--", alpha=0.8, label="VAD-refined .start")

        ax.set_title(f"Window: {win_ms} ms")
        ax.set_ylabel("RMS Energy")

    # Add legend to first subplot (deduplicated), including VAD prob proxy
    import matplotlib.patches as mpatches
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label["VAD prob"] = mpatches.Patch(color="limegreen", alpha=0.4, label="VAD prob")
    axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{meta.title} — {args.lang} energy profile ({args.start}–{args.end}s)", fontsize=13)
    fig.tight_layout()

    out_path = book_dir / "energy_profile.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
