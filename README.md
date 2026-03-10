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

## How it works

1. **Transcription** — Speech-to-text via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with word-level timestamps
2. **Segmentation** — Sentence boundary detection via [pySBD](https://github.com/nipunsadvilkar/pySBD)
3. **Alignment** — Cross-lingual sentence matching using [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings via [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
4. **Assembly** — Audio normalization/extraction/interleaving via [ffmpeg](https://www.ffmpeg.org/)
5. **Metadata** — Cover art + text metadata merging (optionally via local LLM with [ollama](https://ollama.com/))

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on PATH
- CUDA-capable GPU recommended (CPU works but is much slower)

## Installation

```bash
pip install bilbo-audiobook
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install bilbo-audiobook
```

## Usage

### Process a book

```bash
bilbo process \
  --l1-audio en.mp3 \
  --l2-audio de.mp3 \
  --l1-lang en \
  --l2-lang de \
  --title "My Book"
```

This runs the full pipeline (transcribe, segment, align, export) and stores results in `~/.bilbo/books/<slug>/`.

### Library management

```bash
bilbo list              # List all books
bilbo info <slug>       # Show details about a book
bilbo delete <slug>     # Delete a book
```
