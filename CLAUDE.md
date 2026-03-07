# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bilbo is a Python CLI tool that creates bilingual audiobooks by semantically interleaving two audiobooks of the same book in different languages. It uses faster-whisper for transcription, pySBD for sentence segmentation, LaBSE embeddings for cross-lingual alignment, and ffmpeg/soundfile for audio assembly.

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

### Pipeline stages

4-stage sequential pipeline orchestrated by `pipeline.py`, each producing JSON intermediates in `~/.bilbo/books/{slug}/`:

1. **Transcription** (`transcribe.py`) — faster-whisper STT with word-level timestamps → `raw_segments_l{1,2}.json`
2. **Segmentation** (`segment.py`) — Flattens all words, re-segments into sentences via pySBD with char-span mapping → `segments_l{1,2}.json`
3. **Alignment** (`align.py`) — Two-pass algorithm: first finds high-confidence anchor pairs (LaBSE cosine similarity), then fills gaps between anchors with m:n dynamic programming (supports up to 3:3 groupings) → `alignment.json`
4. **Assembly** (`assemble.py`) — Preprocesses audio (resample + LUFS normalize via ffmpeg), extracts chunks per alignment pair, interleaves with gaps/crossfades, streams PCM to ffmpeg encoder → `exports/interleaved.{m4b,mp3,txt}`

### Key modules

- `models.py` — Dataclasses (`Segment`, `Word`, `SegmentedText`, `Alignment`, `AlignmentPair`, `BookMeta`, `ExportConfig`, `ChapterMarker`) with `.save()`/`.load()` JSON serialization
- `audio.py` — Pure audio utilities (slicing, silence, tones, crossfade, fade) + `AudioExporter` context manager that streams PCM chunks to ffmpeg via stdin pipe
- `metadata.py` — ffprobe-based metadata extraction (`SourceMetadata`), cover art extraction/merging (PIL), chapter-to-output timestamp mapping
- `llm.py` — Optional ollama integration (local LLM) for merging L1/L2 metadata text and chapter titles; falls back to `" / "` joining on any error
- `log.py` — `PipelineLog` centralizes all terminal output; `ProgressTracker` for single-line progress, `ParallelTracker` for thread-safe multi-line progress (ANSI cursor management when tty, periodic newlines when piped)
- `library.py` — Book storage at `~/.bilbo/` with `library.json` index; `Library` accepts optional `root` param (used in tests with `tmp_path`)
- `pipeline.py` — Orchestrates the 4 stages, creates `PipelineLog` and passes it down; handles parallel transcription of L1/L2 via ThreadPoolExecutor
- `cli.py` — Click-based CLI entry point; keeps `click.echo` only for non-pipeline output (list, info, delete)

### Lazy imports pattern

Heavy ML dependencies (faster-whisper, sentence-transformers, torch, PIL, mutagen) are imported inside functions rather than at module level. This keeps CLI startup fast and allows tests to run without GPU libraries. `TYPE_CHECKING` guards are used for type hints.

### Progress/logging pattern

Stage modules accept an optional `log: PipelineLog | None` or `on_progress: Callable` parameter. When `None`, they run silently (important for tests). All terminal output flows through `PipelineLog` — stage modules never import `click` for output.

## Conventions

- **Atomic file writes**: write to `.tmp` then rename
- **Skip-if-exists**: pipeline checks for intermediate files; use `--force` to reprocess
- Type hints with `from __future__ import annotations`
- External runtime requirements: `ffmpeg` and `ffprobe` must be on PATH
- Tests use `tmp_library` fixture (`conftest.py`) which creates an isolated `Library(root=tmp_path)` — never touches `~/.bilbo/`
