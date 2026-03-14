from __future__ import annotations

import functools
import shutil
from pathlib import Path

import click

from . import __version__
from .library import Library
from .models import ExportConfig

# Whisper-supported language codes (stable set from openai/whisper)
WHISPER_LANG_CODES = {
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "yue",
    "zh",
}


def _validate_lang(ctx, param, value):
    if value is None:
        return None
    if value not in WHISPER_LANG_CODES:
        raise click.BadParameter(
            f"Unsupported language code '{value}'. "
            f"Use a Whisper-supported code (e.g. en, de, fr, es, zh, ja)."
        )
    return value


def export_options(f):
    """Shared export options for process and export commands."""

    @click.option(
        "--intra-gap",
        type=int,
        default=300,
        help="Gap between L1/L2 within a pair (ms)",
    )
    @click.option("--inter-gap", type=int, default=600, help="Gap between pairs (ms)")
    @click.option(
        "--fade-ms",
        type=int,
        default=15,
        help="Fade duration applied outside speech (ms)",
    )
    @click.option(
        "--format", "fmt", type=click.Choice(["m4b", "mp3"]), default="m4b",
        help="Output audio format"
    )
    @click.option(
        "--no-warn-noise",
        is_flag=True,
        help="Skip warning tones around misaligned regions",
    )
    @click.option(
        "--no-llm-merge", is_flag=True, help="Disable LLM-powered metadata merging"
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def _make_export_config(
    intra_gap: int,
    inter_gap: int,
    fade_ms: int,
    fmt: str,
    no_warn_noise: bool,
    no_llm_merge: bool,
) -> ExportConfig:
    return ExportConfig(
        intra_gap_ms=intra_gap,
        inter_gap_ms=inter_gap,
        fade_ms=fade_ms,
        format=fmt,
        warn_noise=not no_warn_noise,
        llm_merge=not no_llm_merge,
    )



COMMAND_ALIASES = {
    "remove": "delete",
    "run": "process",
}


class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        cmd_name = COMMAND_ALIASES.get(cmd_name, cmd_name)
        return super().get_command(ctx, cmd_name)


@click.group(cls=AliasedGroup)
@click.version_option(__version__, "-v", "--version")
def cli():
    """Bilingual audiobook interleaver."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise click.ClickException(
                f"'{tool}' not found on PATH. "
                f"Please install ffmpeg: https://ffmpeg.org/download.html"
            )


@cli.command(hidden=True)
@click.pass_context
def help(ctx):
    """Show this help message."""
    click.echo(ctx.parent.get_help())


@cli.command()
@click.argument("l1_audio", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.argument("l2_audio", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option(
    "--l1", "l1_lang",
    default=None,
    callback=_validate_lang,
    help="L1 language code (e.g. en); auto-detected if omitted",
)
@click.option(
    "--l2", "l2_lang",
    default=None,
    callback=_validate_lang,
    help="L2 language code (e.g. de); auto-detected if omitted",
)
@click.option(
    "--title",
    default=None,
    help="Book title (auto-generated from filenames if omitted)",
)
@click.option("--from", "from_stage", type=click.IntRange(1, 4), default=None,
              help="Start from this stage (1=transcribe, 2=segment, 3=align, 4=export)")
@click.option("--to", "to_stage", type=click.IntRange(1, 4), default=None,
              help="Stop after this stage (1=transcribe, 2=segment, 3=align, 4=export)")
@click.option("--whisper-model", default="large-v3-turbo", help="Whisper model size")
@click.option("--device", default="auto", help="Compute device (cpu/cuda/auto)")
@export_options
def process(
    l1_audio,
    l2_audio,
    l1_lang,
    l2_lang,
    title,
    from_stage,
    to_stage,
    whisper_model,
    device,
    intra_gap,
    inter_gap,
    fade_ms,
    fmt,
    no_warn_noise,
    no_llm_merge,
):
    """Run the full processing pipeline."""
    from .pipeline import run_pipeline

    # Re-run mode: if --title given and book exists, load audio from stored metadata
    if title is not None:
        lib = Library()
        existing = lib.find(title)
        if existing is not None:
            title = existing.title
            if l1_audio is None:
                l1_audio = Path(existing.l1_audio)
            if l2_audio is None:
                l2_audio = Path(existing.l2_audio)
            if l1_lang is None and existing.l1_lang:
                l1_lang = existing.l1_lang
            if l2_lang is None and existing.l2_lang:
                l2_lang = existing.l2_lang

    if (l1_audio is None) != (l2_audio is None):
        raise click.ClickException("Provide both audio paths or neither.")
    if l1_audio is None:
        raise click.ClickException(
            "Missing audio paths. "
            "Pass both L1 and L2 audio files, or --title of an existing book to re-run."
        )

    config = _make_export_config(
        intra_gap, inter_gap, fade_ms, fmt, no_warn_noise, no_llm_merge
    )
    meta = run_pipeline(
        l1_audio=l1_audio,
        l2_audio=l2_audio,
        l1_lang=l1_lang,
        l2_lang=l2_lang,
        title=title,
        model_size=whisper_model,
        device=device,
        export_config=config,
        from_stage=from_stage,
        to_stage=to_stage,
    )
    click.echo(f"\nBook '{meta.title}' saved.")


@cli.command("list")
def list_cmd():
    """List all books in the library."""
    lib = Library()
    books = lib.list_books()
    if not books:
        click.echo("Library is empty.")
        return
    for i, book in enumerate(books, 1):
        stages = ",".join(str(s) for s in sorted(book.stages_completed))
        exports = ", ".join(book.exports) if book.exports else "none"
        click.echo(
            f"  {i}. {book.title} [{book.l1_lang}/{book.l2_lang}] stages=[{stages}] exports=[{exports}]"
        )


@cli.command()
@click.argument("title")
def info(title):
    """Show details about a book."""
    lib = Library()
    meta = lib.find(title)
    if meta is None:
        raise click.ClickException(f"Book '{title}' not found.")
    click.echo(f"Title:    {meta.title}")
    if meta.author:
        click.echo(f"Author:   {meta.author}")
    click.echo(f"L1:       {meta.l1_lang} ({meta.l1_audio})")
    click.echo(f"L2:       {meta.l2_lang} ({meta.l2_audio})")
    click.echo(f"Stages:   {sorted(meta.stages_completed)}")
    click.echo(f"Exports:  {meta.exports or 'none'}")


@cli.command()
@click.argument("title")
@click.argument("new_title")
def rename(title, new_title):
    """Rename a book."""
    lib = Library()
    meta = lib.rename(title, new_title)
    if meta is None:
        raise click.ClickException(f"Book '{title}' not found.")
    click.echo(f"Renamed '{title}' -> '{new_title}'")


@cli.command()
@click.argument("title")
@click.confirmation_option(prompt="Are you sure you want to delete this book?")
def delete(title):
    """Delete a book from the library."""
    lib = Library()
    meta = lib.find(title)
    if meta is None:
        raise click.ClickException(f"Book '{title}' not found.")
    lib.delete(meta.slug)
    click.echo(f"Deleted '{title}'.")


