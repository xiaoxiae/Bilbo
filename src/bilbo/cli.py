from __future__ import annotations

from pathlib import Path

import click

from .library import Library
from .models import ExportConfig


@click.group()
@click.version_option(package_name="bilbo")
def cli():
    """Bilingual audiobook interleaver."""
    pass


@cli.command()
@click.option("--l1-audio", required=True, type=click.Path(exists=True, path_type=Path), help="L1 audiobook file")
@click.option("--l2-audio", required=True, type=click.Path(exists=True, path_type=Path), help="L2 audiobook file")
@click.option("--l1-lang", required=True, help="L1 language code (e.g. en)")
@click.option("--l2-lang", required=True, help="L2 language code (e.g. de)")
@click.option("--title", required=True, help="Book title")
@click.option("--intra-gap", type=int, default=300, help="Gap between L1/L2 within a pair (ms)")
@click.option("--inter-gap", type=int, default=600, help="Gap between pairs (ms)")
@click.option("--format", "fmt", type=click.Choice(["m4b", "mp3"]), default="m4b")
@click.option("--whisper-model", default="large-v3-turbo", help="Whisper model size")
@click.option("--device", default="auto", help="Compute device (cpu/cuda/auto)")
@click.option("--order", type=click.Choice(["l1-first", "l2-first"]), default="l1-first")
@click.option("--no-export", is_flag=True, help="Stop after alignment, skip audio export")
@click.option("--force", is_flag=True, help="Re-run all stages")
@click.option("--batch-size", type=int, default=None, help="Whisper batch size (default: 16)")
def process(l1_audio, l2_audio, l1_lang, l2_lang, title, intra_gap, inter_gap, fmt, whisper_model, device, order, no_export, force, batch_size):
    """Run the full processing pipeline."""
    from .pipeline import run_pipeline

    config = ExportConfig(intra_gap_ms=intra_gap, inter_gap_ms=inter_gap, format=fmt, order=order)
    meta = run_pipeline(
        l1_audio=l1_audio,
        l2_audio=l2_audio,
        l1_lang=l1_lang,
        l2_lang=l2_lang,
        title=title,
        model_size=whisper_model,
        device=device,
        no_export=no_export,
        export_config=config,
        force=force,
        batch_size=batch_size,
    )
    click.echo(f"\nBook '{meta.title}' saved as '{meta.slug}'.")


@cli.command("export")
@click.argument("slug")
@click.option("--intra-gap", type=int, default=300, help="Gap between L1/L2 within a pair (ms)")
@click.option("--inter-gap", type=int, default=600, help="Gap between pairs (ms)")
@click.option("--format", "fmt", type=click.Choice(["m4b", "mp3"]), default="m4b")
@click.option("--order", type=click.Choice(["l1-first", "l2-first"]), default="l1-first")
def export_cmd(slug, intra_gap, inter_gap, fmt, order):
    """Export an interleaved audiobook from an already-processed book."""
    from .pipeline import run_export

    config = ExportConfig(intra_gap_ms=intra_gap, inter_gap_ms=inter_gap, format=fmt, order=order)
    run_export(slug, config)


@cli.command("list")
def list_cmd():
    """List all books in the library."""
    lib = Library()
    books = lib.list_books()
    if not books:
        click.echo("Library is empty.")
        return
    for book in books:
        stages = ",".join(str(s) for s in sorted(book.stages_completed))
        exports = ", ".join(book.exports) if book.exports else "none"
        click.echo(f"  {book.slug}: {book.title} [{book.l1_lang}/{book.l2_lang}] stages=[{stages}] exports=[{exports}]")


@cli.command()
@click.argument("slug")
def info(slug):
    """Show details about a book."""
    lib = Library()
    meta = lib.get(slug)
    if meta is None:
        raise click.ClickException(f"Book '{slug}' not found.")
    click.echo(f"Title:    {meta.title}")
    click.echo(f"Slug:     {meta.slug}")
    click.echo(f"L1:       {meta.l1_lang} ({meta.l1_audio})")
    click.echo(f"L2:       {meta.l2_lang} ({meta.l2_audio})")
    click.echo(f"Stages:   {sorted(meta.stages_completed)}")
    click.echo(f"Exports:  {meta.exports or 'none'}")


@cli.command()
@click.argument("slug")
@click.confirmation_option(prompt="Are you sure you want to delete this book?")
def delete(slug):
    """Delete a book from the library."""
    lib = Library()
    if lib.delete(slug):
        click.echo(f"Deleted '{slug}'.")
    else:
        raise click.ClickException(f"Book '{slug}' not found.")
