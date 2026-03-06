from __future__ import annotations

import sys
import threading

import click


class ProgressTracker:
    """Single in-place progress line (e.g. transcribing, assembling)."""

    def __init__(self, description: str, unit: str, log: PipelineLog) -> None:
        self._desc = description
        self._unit = unit
        self._log = log
        self._last_pct = -1

    def _format(self, current: float, total: float | None) -> str:
        if self._unit:
            if total and total > 0:
                pct = min(100.0, current / total * 100)
                return f"  {self._desc}: {pct:5.1f}% ({current:.0f}/{total:.0f}{self._unit})"
            return f"  {self._desc}: {current:.0f}{self._unit}"
        else:
            if total and total > 0:
                return f"  {self._desc}:  {int(current)}/{int(total)}"
            return f"  {self._desc}:  {int(current)}"

    def update(self, current: float, total: float | None = None) -> None:
        if self._log._is_tty:
            click.echo(f"\r\033[K{self._format(current, total)}", nl=False)
            self._log._needs_newline = True
        else:
            if total and total > 0:
                pct = int(current / total * 10)
                if pct != self._last_pct:
                    self._last_pct = pct
                    click.echo(self._format(current, total))

    def finish(self, msg: str) -> None:
        if self._log._is_tty:
            click.echo(f"\r\033[K  {click.style('\u2713', fg='green')} {msg}")
        else:
            click.echo(f"  \u2713 {msg}")
        self._log._needs_newline = False


class ParallelTracker:
    """Thread-safe multi-line progress (e.g. parallel preprocessing)."""

    def __init__(
        self, labels: list[str], description: str, unit: str, log: PipelineLog
    ) -> None:
        self._labels = labels
        self._desc = description
        self._unit = unit
        self._log = log
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float | None]] = {}
        self._lines_printed = 0
        self._last_pcts: dict[str, int] = {}

    def _format_line(self, label: str) -> str:
        if label in self._state:
            current, total = self._state[label]
            if total and total > 0:
                pct = min(100.0, current / total * 100)
                if self._unit:
                    return f"  [{label}] {self._desc}: {pct:5.1f}% ({current:.0f}/{total:.0f}{self._unit})"
                return f"  [{label}] {self._desc}: {pct:5.1f}% ({current:.0f}/{total:.0f})"
            if self._unit:
                return f"  [{label}] {self._desc}: {current:.0f}{self._unit}"
            return f"  [{label}] {self._desc}: {current:.0f}"
        return f"  [{label}] {self._desc}: waiting..."

    def update(self, label: str, current: float, total: float | None = None) -> None:
        with self._lock:
            self._state[label] = (current, total)
            if self._log._is_tty:
                n = len(self._labels)
                if self._lines_printed == 0:
                    for i, lbl in enumerate(self._labels):
                        click.echo(
                            f"\033[K{self._format_line(lbl)}",
                            nl=(i < n - 1),
                        )
                    self._lines_printed = n
                else:
                    if n > 1:
                        click.echo(f"\r\033[{n - 1}A", nl=False)
                    else:
                        click.echo("\r", nl=False)
                    for i, lbl in enumerate(self._labels):
                        click.echo(
                            f"\033[K{self._format_line(lbl)}",
                            nl=(i < n - 1),
                        )
                self._log._needs_newline = True
            else:
                if total and total > 0:
                    pct = int(current / total * 10)
                    if pct != self._last_pcts.get(label, -1):
                        self._last_pcts[label] = pct
                        click.echo(self._format_line(label))

    def finish(self, msg: str) -> None:
        with self._lock:
            if self._log._is_tty and self._lines_printed > 0:
                n = len(self._labels)
                if n > 1:
                    click.echo(
                        f"\r\033[{n - 1}A\033[J  {click.style('\u2713', fg='green')} {msg}"
                    )
                else:
                    click.echo(
                        f"\r\033[K  {click.style('\u2713', fg='green')} {msg}"
                    )
            else:
                click.echo(f"  \u2713 {msg}")
            self._log._needs_newline = False

    def callback(self, label: str):
        """Return a 2-arg callback bound to *label* for use from worker threads."""

        def _cb(current: float, total: float | None = None) -> None:
            self.update(label, current, total)

        return _cb


class PipelineLog:
    """Centralized progress reporting for the bilbo pipeline."""

    def __init__(self) -> None:
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._needs_newline = False

    def _close_line(self) -> None:
        if self._needs_newline:
            click.echo()
            self._needs_newline = False

    def stage(self, num: int, title: str) -> None:
        self._close_line()
        header = f"\u2500\u2500 Stage {num}: {title} "
        header = header.ljust(50, "\u2500")
        click.echo(click.style(header, bold=True))

    def skip(self, msg: str) -> None:
        self._close_line()
        click.echo(click.style(f"  \u23ed {msg}", dim=True))

    def info(self, msg: str) -> None:
        self._close_line()
        click.echo(f"  {msg}")

    def done(self, msg: str) -> None:
        self._close_line()
        click.echo(f"  {click.style('\u2713', fg='green')} {msg}")

    def warn(self, msg: str) -> None:
        self._close_line()
        click.echo(click.style(f"  \u26a0 {msg}", fg="yellow"))

    def progress(self, description: str, unit: str = "") -> ProgressTracker:
        self._close_line()
        return ProgressTracker(description, unit, self)

    def parallel(
        self, labels: list[str], description: str, unit: str = "s"
    ) -> ParallelTracker:
        self._close_line()
        return ParallelTracker(labels, description, unit, self)
