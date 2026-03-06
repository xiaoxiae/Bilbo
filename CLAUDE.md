# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bilbo is a Python CLI tool that creates bilingual audiobooks by semantically interleaving two audiobooks of the same book in different languages. It uses Whisper for transcription, pySBD for sentence segmentation, LaBSE embeddings for cross-lingual alignment, and ffmpeg/soundfile for audio assembly.

## Commands

```bash
# Install dependencies
uv sync

# Run CLI
bilbo process --l1-audio en.mp3 --l2-audio de.mp3 --l1-lang en --l2-lang de --title "Book"
bilbo export <slug>
bilbo list
bilbo info <slug>
bilbo delete <slug>

# Tests
uv run pytest                        # all tests
uv run pytest tests/test_segment.py  # single file
```

## Architecture

The tool implements a 4-stage sequential pipeline, each producing JSON intermediates in `~/.bilbo/books/{slug}/`:

1. **Transcription** (`transcribe.py`) — Whisper STT with word-level timestamps → `raw_segments_l{1,2}.json`
2. **Segmentation** (`segment.py`) — Re-segments into proper sentences via pySBD, mapping word timestamps → `segments_l{1,2}.json`
3. **Alignment** (`align.py`) — LaBSE embeddings + dynamic programming to match L1↔L2 sentence groups (supports 1:1, 1:2, 2:1, 1:3, 3:1, 2:2) → `alignment.json`
4. **Assembly** (`assemble.py`) — Extracts audio chunks via ffmpeg preprocessing, applies crossfades, adds silence gaps → `interleaved.m4b`/`.mp3`

**Key modules:**
- `models.py` — Dataclasses (`Segment`, `SegmentedText`, `Alignment`, `BookMeta`, `ExportConfig`) with `.save()`/`.load()` JSON serialization
- `audio.py` — Pure audio utility: slicing, silence generation, crossfading, ffmpeg preprocessing (no terminal output)
- `log.py` — `PipelineLog` centralizes all terminal output; `ProgressTracker` for single-line progress, `ParallelTracker` for thread-safe multi-line progress (ANSI cursor management when tty, periodic newlines when piped)
- `library.py` — Book storage and `library.json` index management
- `pipeline.py` — Orchestrates the 4 stages, creates `PipelineLog` and passes it down
- `cli.py` — Click-based CLI entry point; keeps `click.echo` only for non-pipeline output (list, info, delete)

### Progress/logging pattern

Stage modules (`transcribe`, `segment`, `align`, `assemble`) accept an optional `log: PipelineLog | None` or `on_progress: Callable` parameter. When `None`, they run silently (important for tests). All terminal output flows through `PipelineLog` — stage modules never import `click` for output.

## Conventions

- **Atomic file writes**: write to `.tmp` then rename
- **Skip-if-exists**: pipeline checks for intermediate files; use `--force` to reprocess
- Type hints with `from __future__ import annotations`
- Library stored at `~/.bilbo/books/{slug}/`
