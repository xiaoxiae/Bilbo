# Bilingual Audiobook Interleaver

**CLI Tool · Personal Use · GPL-3.0**

## What It Is

A command-line tool that takes two audiobooks of the same book in different languages (e.g. English and German), aligns them semantically, and produces an interleaved audiobook that alternates between languages at a configurable granularity: sentence, paragraph, page, or chapter. The primary use case is passive language learning: you hear a chunk in your stronger language, then immediately hear the equivalent chunk in the language you're learning. Sentence-level interleaving is the most granular and best for close study; chapter-level is closest to a normal listening experience and better for comprehension practice.

## Inputs and Outputs

### Inputs

| Input | Description |
|-------|-------------|
| Audio file (L1) | Audiobook in language 1 (mp3, m4b, flac, etc.) |
| Audio file (L2) | Audiobook in language 2, same book |
| Text (L1) | Optional. Transcript or ebook text in language 1. If not provided, generated via Whisper STT. |
| Text (L2) | Optional. Transcript or ebook text in language 2. If not provided, generated via Whisper STT. |

### Outputs

| Output | Description |
|--------|-------------|
| Alignment JSON | List of matched sentence pairs with timestamps and text for both languages. |
| Interleaved audio | Single audio file (mp3/m4b) alternating between languages at the chosen granularity (sentence/paragraph/page/chapter). Includes chapter markers and metadata. |

## The Pipeline

The tool runs as a sequential pipeline with four stages. Each stage produces an intermediate file so any stage can be re-run or manually corrected without redoing earlier work.

### Stage 1: Transcription

**Tool:** Whisper (large-v3-turbo checkpoint) via faster-whisper

Each audio file is transcribed independently, producing a list of segments with start/end timestamps and text. If the user provides text (e.g. from an ebook), this stage is skipped for that language and the text is used directly, requiring only forced alignment against the audio to obtain timestamps.

**Output:** Two JSON files, one per language, each containing a list of `{start, end, text}` segments.

### Stage 2: Text Segmentation

**Tool:** spaCy or pySBD (sentences), heuristics (paragraphs, pages, chapters)

Whisper segments don't always correspond to clean linguistic boundaries. The raw transcript text is concatenated and re-segmented at multiple granularity levels:

- **Sentence:** The text is re-split into proper sentences using an NLP sentence splitter. Sentence timestamps are interpolated from the original Whisper segment timestamps by character offset proportion.
- **Paragraph:** Sentences are grouped into paragraphs. When source text is provided (ebook/transcript), paragraph breaks come directly from the text. When working from Whisper output only, paragraphs are inferred by detecting longer silences between segments (configurable threshold, default 1.5s) or by using a fixed sentence count (e.g. every 5 sentences).
- **Page:** Relevant only when source text with page markers is provided (e.g. from a PDF or epub with page numbers). Pages are mapped to timestamp ranges via the underlying sentence alignment. When page information is unavailable, this mode falls back to a configurable duration-based split (default ~2 minutes per chunk).
- **Chapter:** Chapters are detected from source audio chapter markers (m4b), ebook chapter boundaries, or large silence gaps in the audio (configurable threshold, default 5s). This is the coarsest granularity.

All levels are derived from the sentence-level segmentation — sentences are the atomic unit, and paragraphs/pages/chapters are groupings of sentences. This means the alignment (Stage 3) always operates at sentence level regardless of the chosen interleaving granularity.

**Output:** Two JSON files per language containing sentences with timestamps, plus paragraph/page/chapter boundary indices: `{sentences: [{start, end, text}, ...], paragraphs: [0, 5, 12, ...], chapters: [0, 48, 103, ...]}`. The boundary arrays store the sentence index where each unit begins.

### Stage 3: Bilingual Alignment

**Tool:** Bertalign (GPL-3.0, uses LaBSE sentence embeddings)

The two sentence lists are fed into Bertalign, which uses multilingual sentence embeddings to compute a similarity matrix and then runs a two-step dynamic programming algorithm. Step one finds 1:1 anchor points; step two refines the alignment within anchor windows, producing 1:N, N:1, and N:M mappings as needed.

This is the core of the tool. No LLM is required. Bertalign handles the semantic matching entirely through pre-trained multilingual embeddings, which natively understand that sentences in different languages can express the same meaning.

**Output:** An alignment JSON file — a list of pairs, where each pair contains one or more sentences from each language with their timestamps:

```json
[
  {
    "l1": [{"start": 0.0, "end": 2.3, "text": "Two years later."}],
    "l2": [{"start": 0.0, "end": 2.8, "text": "Zwei Jahre später."}]
  },
  {
    "l1": [{"start": 2.3, "end": 5.1, "text": "The mountains were covered in snow."}],
    "l2": [
      {"start": 2.8, "end": 4.0, "text": "Die Berge waren schneebedeckt."},
      {"start": 4.0, "end": 5.5, "text": "Es war bitterkalt."}
    ]
  }
]
```

### Stage 4: Audio Assembly

**Tool:** ffmpeg / pydub

The alignment from Stage 3 is grouped according to the chosen interleaving granularity, then the corresponding audio segments are extracted and concatenated in alternating order.

**How granularity affects assembly:**

- **Sentence** (`--interleave sentence`): Each aligned sentence pair is interleaved individually: L1 sentence → gap → L2 sentence → gap → next pair. Most granular, best for close study. Produces the longest output (~2.2x original length with gaps).
- **Paragraph** (`--interleave paragraph`): Aligned sentences are grouped by paragraph boundaries. All L1 sentences in a paragraph play contiguously, then all corresponding L2 sentences, then the next paragraph. A natural middle ground.
- **Page** (`--interleave page`): Same logic as paragraph but grouped by page boundaries. Each page's worth of L1 audio plays, then the corresponding L2 audio.
- **Chapter** (`--interleave chapter`): The coarsest mode. An entire chapter plays in L1, then the same chapter plays in L2. Closest to a normal audiobook experience. Produces the shortest output (~2.0x original length, minimal gap overhead).

Note that because paragraph/page/chapter boundaries will rarely align perfectly between languages (different narrators, different editions), the tool uses the L1 boundaries as the primary structure and groups L2 sentences accordingly based on the sentence-level alignment.

Key audio processing considerations:

- Volume normalization between the two audiobooks (target -16 LUFS, the audiobook standard).
- Short crossfades (20–50ms) at each cut point to avoid clicks and pops.
- Configurable silence gap between languages (default: 300ms for sentence mode, 1s for paragraph, 2s for page/chapter).
- Chapter markers in the output file: in sentence/paragraph/page mode, chapters are carried over from source; in chapter mode, each L1+L2 chapter pair becomes a chapter in the output.
- Sample rate matching (resample to the higher of the two sources).

**Output:** A single audio file (m4b with chapter metadata, or mp3) ready for offline listening.

## The Library

Processed books are stored in a local library directory with a simple flat structure:

```
~/.bilingual-audiobooks/
  library.json                    # index of all books
  books/
    the-trial/
      meta.json                   # title, authors, languages, date processed
      segments_l1.json            # Stage 1+2 output (sentences + boundary indices)
      segments_l2.json
      alignment.json              # Stage 3 output (sentence-level, always)
      exports/
        sentence.m4b              # Stage 4 outputs (one per granularity requested)
        paragraph.m4b
```

The `library.json` index allows the CLI to list, search, and manage books without reading each subdirectory. Multiple exports at different granularities can coexist since the alignment is always sentence-level and grouping happens at export time.

## CLI Interface

### Core Commands

```bash
# Full pipeline: transcribe, align, and export (sentence-level by default)
bili process --l1-audio book_en.mp3 --l2-audio book_de.mp3 \
             --l1-lang en --l2-lang de --title "The Trial"

# Paragraph-level interleaving
bili process --l1-audio book_en.mp3 --l2-audio book_de.mp3 \
             --l1-lang en --l2-lang de --title "The Trial" \
             --interleave paragraph

# With optional text input (skips STT for that language)
bili process --l1-audio book_en.mp3 --l1-text book_en.txt \
             --l2-audio book_de.mp3 --l2-lang de

# Re-run only the export stage (e.g. to change granularity or gap duration)
bili export the-trial --interleave chapter --gap 2000 --format mp3

# Library management
bili list
bili info the-trial
bili delete the-trial
```

### Key Options

| Flag | Purpose |
|------|---------|
| `--interleave <mode>` | Interleaving granularity: `sentence` (default), `paragraph`, `page`, or `chapter` |
| `--gap <ms>` | Silence between languages (default varies by mode: 300/1000/2000ms) |
| `--format <fmt>` | Output format: m4b (default, with chapters) or mp3 |
| `--whisper-model` | Whisper model size (default: large-v3-turbo) |
| `--device` | Compute device for Whisper and embeddings (cpu/cuda) |
| `--order` | Playback order: l1-first (default) or l2-first |
| `--no-export` | Stop after alignment, don't produce audio |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| STT | OpenAI Whisper (large-v3-turbo) via faster-whisper |
| Sentence splitting | spaCy (with per-language model) or pySBD |
| Bilingual alignment | Bertalign (LaBSE embeddings, DP alignment) |
| Audio processing | ffmpeg via pydub, pyloudnorm for normalization |
| CLI framework | click or typer |
| License | GPL-3.0 (driven by Bertalign dependency) |

## Known Risks and Mitigations

### Alignment quality

Bertalign achieves ~94% strict precision and ~99% lax precision on benchmark data, but audiobook transcripts are noisier than clean text. Whisper transcription errors, especially for proper nouns or unusual vocabulary, may degrade embedding similarity. Mitigation: provide text input when available, and expose a confidence score per pair so low-quality alignments can be flagged and manually reviewed.

### Sentence boundary mismatch

Different narrators pace and phrase sentences differently. One narrator might pause mid-sentence where another doesn't, causing Whisper to split differently. The sentence re-segmentation stage (Stage 2) mitigates this, but imperfect timestamp interpolation may cause cuts slightly before or after the actual sentence. A small padding buffer (50–100ms) on each side of each segment handles this gracefully.

### Paragraph and chapter detection without source text

When working from Whisper output alone (no ebook text provided), paragraph and chapter boundaries must be inferred from silence gaps. This is inherently heuristic — a long pause might be a paragraph break, a dramatic pause, or just the narrator breathing. The tool uses configurable thresholds and allows the user to re-export with different settings without re-running alignment. When source text is available, paragraph and chapter boundaries are much more reliable.

### Output file size

The interleaved audiobook is roughly 2x the length of either original, plus silence gaps. At sentence-level interleaving, a 10-hour audiobook becomes ~22 hours due to the many gaps. At chapter level the overhead is minimal (~20 hours). This is inherent to the approach and cannot be avoided, but the user should be warned during processing.

### Long processing time

Whisper transcription of two full audiobooks is the bottleneck, potentially taking several hours on CPU. GPU acceleration (CUDA) reduces this dramatically. The pipeline saves intermediate results, so a crash or interruption doesn't lose progress.

## Future Possibilities

These are explicitly out of scope for v1 but worth keeping in mind for the architecture:

- Web player with synchronized text highlighting and pair-by-pair navigation.
- Adjustable playback modes: L2-only with L1 on demand, repeat L2 twice, speed control per language.
- Confidence-based manual review UI for fixing misaligned pairs.
- Support for more than two languages (e.g. triangulating EN/DE/FR of the same book).
- Integration with Audiobookshelf or similar self-hosted audiobook servers.
