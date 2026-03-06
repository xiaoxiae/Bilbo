# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bilbo is a Python CLI tool that creates bilingual audiobooks by semantically interleaving two audiobooks of the same book in different languages. It uses Whisper for transcription, pySBD for sentence segmentation, LaBSE embeddings for cross-lingual alignment, and pydub/pyloudnorm for audio assembly.

## Commands

```bash
# Install dependencies
uv sync

# Run CLI
bilbo process --l1-audio en.mp3 --l2-audio de.mp3 --l1-lang en --l2-lang de --title "Book"
bilbo export <slug> --interleave chapter
bilbo list
bilbo info <slug>
bilbo delete <slug>

# Tests
pytest                        # all tests
pytest tests/test_segment.py  # single file

# Lint (ruff is configured but not in dev deps)
ruff check src/
```

## Architecture

The tool implements a 4-stage sequential pipeline, each producing JSON intermediates in `~/.bilbo/books/{slug}/`:

1. **Transcription** (`transcribe.py`) — Whisper STT with word-level timestamps → `raw_segments_l{1,2}.json`
2. **Segmentation** (`segment.py`) — Re-segments into proper sentences via pySBD, mapping word timestamps → `segments_l{1,2}.json`
3. **Alignment** (`align.py`) — LaBSE embeddings + dynamic programming to match L1↔L2 sentence groups (supports 1:1, 1:2, 2:1, 1:3, 3:1, 2:2) → `alignment.json`
4. **Assembly** (`assemble.py`) — Extracts audio chunks, normalizes loudness (target -16 LUFS), applies crossfades, adds silence gaps → `interleaved.m4b`/`.mp3`

**Key modules:**
- `models.py` — Dataclasses (`Segment`, `SegmentedText`, `Alignment`, `BookMeta`, `ExportConfig`) with `.save()`/`.load()` JSON serialization
- `audio.py` — Audio I/O, slicing, silence generation, crossfading, LUFS normalization (ffmpeg fallback for loading)
- `library.py` — Book storage and `library.json` index management
- `pipeline.py` — Orchestrates the 4 stages
- `cli.py` — Click-based CLI entry point

## Conventions

- **Atomic file writes**: write to `.tmp` then rename
- **Skip-if-exists**: pipeline checks for intermediate files; use `--force` to reprocess
- Type hints with `from __future__ import annotations`
- Library stored at `~/.bilbo/books/{slug}/`
