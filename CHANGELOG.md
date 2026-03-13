# Changelog

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
