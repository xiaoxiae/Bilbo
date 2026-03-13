# Bilbo

Bilingual audiobook interleaver. Takes **two audiobooks** of the same book in **different languages** and creates a single audiobook that **alternates between them sentence-by-sentence.**

## Example

From *The Alloy of Law* by Brandon Sanderson ([English](https://www.audible.com/pd/The-Alloy-of-Law-Audiobook/B005ZUI3OA) + [German](https://www.audible.com/pd/Hueter-des-Gesetzes-Audiobook/B0CGF5XX82)):

https://github.com/user-attachments/assets/aff6c1cb-6f67-43a8-bc9f-3fd6ab3a5c8a

> **EN:** The revolver was nothing fancy to look at, though the six-shot cylinder was machined with such care in the steel alloy frame that there was no play in its movement.
>
> **DE:** Der Revolver machte zwar keinen besonders ansehnlichen Eindruck, doch die sechsschüssige Trommel war mit solcher Präzision in den Rahmen aus einer Stahllegierung eingesetzt, dass in ihren Bewegungen nicht das geringste Spiel war.

---

> **EN:** There was no gleam to the metal or exotic material on the grip, but it fit his hand like it was meant to be there.
>
> **DE:** Das Metall schimmerte nicht und in den Griff waren keinerlei exotische Materialien eingelassen, aber die Waffe lag so gut in seiner Hand, als wäre sie eigens dafür geschaffen worden.

---

> **EN:** The waist-high fence was flimsy, the wood grayed with time, held together with fraying lengths of rope.
>
> **DE:** Der hüfthohe Zaun war baufällig, das Holz, mit der Zeit grau geworden, wurde von ausgefransten Seilen zusammengehalten.

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on PATH
- CUDA-capable GPU recommended (CPU works but is much slower)

## Installation

```bash
pip install bilbo-audiobook
```

or CPU-only as

```bash
pip install bilbo-audiobook[cpu]
```

## Usage

### Process a book

To process an entire book from start to finish, run


```bash
bilbo process data/en-5min.m4a data/de-5min.m4a --title "My Book"
```

which runs the full pipeline (transcribe, segment, align, export) and stores results in `~/.bilbo/books/<title>/`.

```output
── Stage 0: Input ────────────────────────────────
  ✓ Input audio copied

── Stage 1: Transcription ────────────────────────
  ✓ Model loaded  (large-v3-turbo, cuda)
  ✓ L1: 10 segments, L2: 13 segments
  Detected L1 language: en
  Detected L2 language: de

── Stage 2: Segmentation ─────────────────────────
  ✓ EN: 63 sentences, DE: 64 sentences
    EN: extended 54/63 ends (avg +141ms)
    DE: extended 57/64 ends (avg +80ms)

── Stage 3: Alignment ────────────────────────────
  ✓ LaBSE model loaded  (cuda)
  ✓ Embeddings computed
  ⠋ Filling gaps...
  ✓ 30 anchors
  ✓ 55 pairs

── Stage 4: Assembly ─────────────────────────────
  ✓ Metadata extracted
    Titles: The Alloy of Law: A Mistborn Novel / Hüter des Gesetzes: Mistborn 4
    Artists: Brandon Sanderson / Brandon Sanderson, Michael Siefener - Übersetzer
    Chapters: EN=1, DE=2
    Cover art: both sources
  ✓ Preprocessed (EN: 295s, DE: 343s)
  ⠋ Assembling...
  ✓ Metadata merged via LLM
    comment: Three hundred years after the events of the Mistborn trilogy, Scadrial is now on the verge of modernity. Yet the old magics of Allomancy and Feruchemy continue to play a role in this reborn world....
    title: The Alloy of Law: A Mistborn Novel / Hüter des Gesetzes: Mistborn 4
    artist: Brandon Sanderson
    album: The Alloy of Law (Unabridged) / Hüter des Gesetzes: Mistborn 4
  ✓ 10.5 minutes
  ✓ 1 output chapters

Done in 28.5s  (Stage 0: 0.0s, Stage 1: 4.8s, Stage 2: 0.2s, Stage 3: 5.8s, Stage 4: 16.2s)

Book 'My Book' saved.
```

If you're running on CPU only, this will take a **VERY** long time, unless you're running a short snippet.

### Library management

```bash
bilbo list                          # List all books
bilbo info <title>                  # Show details about a book
bilbo rename <title> "New Title"    # Rename a book
bilbo delete <title>                # Delete a book
```

## How it works

1. **Transcription** — Speech-to-text via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with word-level timestamps
2. **Segmentation** — Sentence boundary detection via [pySBD](https://github.com/nipunsadvilkar/pySBD)
3. **Alignment** — Cross-lingual sentence matching using [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings via [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
4. **Assembly** — Audio normalization/extraction/interleaving via [ffmpeg](https://www.ffmpeg.org/)
5. **Metadata** — Cover art + text metadata merging (optionally via local LLM with [ollama](https://ollama.com/))
