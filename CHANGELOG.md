# Changelog

## [0.1.4] - 2026-03-17

### Added
- Numeric IDs in `bilbo list` output; all commands now accept a numeric ID instead of a title (e.g., `bilbo info 1`) (f9138b0)
- Command aliases: `remove` → `delete`, `run` → `process` (f9138b0)
- `bilbo help` command (89936d2)
- Live VAD progress (`EN VAD: 47%`) during segmentation stage using Silero's `progress_tracking_callback` (bb266b8)

### Changed
- `bilbo process` audio arguments are now optional to allow re-running stages via `--from`/`--to` without re-specifying audio files (89936d2)
- Removed per-stage CLI commands (`bilbo transcribe`, `bilbo segment`, `bilbo align`, `bilbo export`) in favour of `bilbo process --from N --to N` (89936d2)
- Default Whisper model changed from `large-v3-turbo` to `large-v3` (4ce3820)
- Transcription now runs sequentially (L1 then L2) rather than in parallel; a single shared model cannot run concurrently (4ce3820)
- `refine_timestamps` now uses ownership-based VAD matching: a speech region is assigned to the segment that contains >50% of its duration, and both `.start` and `.end` are snapped to the matched regions' extremes with ±50 ms padding (bb266b8)

### Fixed
- Assembly no longer redundantly reopens audio files for each pair (23c6729)

## [0.1.3] - 2026-03-14

### Added
- Ported full Bertalign two-pass alignment algorithm (overlap encoding, anchor DP, m:n second pass with margin scoring) replacing the previous anchor+gap-fill approach (2bd8bc1)
- GitHub Actions release workflow: auto-publish to PyPI on tag, TestPyPI `.devN` builds on main push (4c79154)
- Log skipped sentences during alignment (98aab78)

### Changed
- Cover merging uses vectorized numpy instead of per-row loop (7118f92)
- `refine_timestamps` reuses a single open `SoundFile` instead of re-reading per segment (7118f92)
- `find_problematic_regions` uses numpy cumsum for sliding window (7118f92)
- Deduplicated chapter-finishing logic in `map_chapters_to_output` (7118f92)
- Fixed `Library.rename` to remap audio paths by directory component, not string replace (7118f92)

## [0.1.2] - 2026-03-13

### Added
- Language auto-detection (omit `--l1`/`--l2` to detect from audio) (6080564)
- `bilbo rename` command to rename books (6080564)
- `bilbo transcribe`, `bilbo segment`, `bilbo align` per-stage commands (6080564)
- Slug-based filesystem paths (titles can now contain spaces and special characters) (6080564)

### Changed
- CLI now accepts human-readable titles instead of slugs (6080564)
- Library index keyed by auto-generated slugs internally (6080564)

### Fixed
- `--device auto` now correctly resolves to CUDA/CPU (f9f2ec8)

## [0.1.1] - 2026-03-12

### Added
- LLM-powered metadata merging (via local ollama) (86f3ff2)
- Diagonal poster/cover merging (d8e80a0)
- Warning tones around misaligned regions (d3d1ac3)
- Better problem passage detection with sliding-window smoothing (d3d1ac3)
- Energy-based segment timestamp extension (d64d16e)
- PyPI release (`pip install bilbo-audiobook`) (8b0dfdf)

### Changed
- Improved logging and progress reporting (4afcf26)
- Refactored CLI with Click (d64d16e)

### Fixed
- Metadata no longer lost on re-export (ab003a0)

## [0.1.0] - 2026-03-10

### Added
- Initial release (dfce168)
- 4-stage pipeline: transcription, segmentation, alignment, assembly (dfce168)
- faster-whisper transcription with word-level timestamps (dfce168)
- pySBD sentence segmentation (dfce168)
- LaBSE cross-lingual alignment (anchor + gap-fill DP) (7475dbe)
- Audio assembly with LUFS normalization, gaps, and crossfades (38f8bee)
- Text export of aligned pairs (dd051fa)
- Metadata extraction and cover art merging (a6ad026)
- Multi-threaded parallel transcription (38f8bee)
- Library management (`bilbo list`, `bilbo info`, `bilbo delete`) (dfce168)
