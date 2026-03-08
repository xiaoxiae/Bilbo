from __future__ import annotations

import sys
import threading
import time

import click

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ProgressTracker:
    """Single in-place progress line (e.g. transcribing, assembling)."""

    def __init__(self, description: str, unit: str, log: PipelineLog) -> None:
        self._desc = description
        self._unit = unit
        self._log = log
        self._last_pct = -1
        self._spin_idx = 0
        self._current: float = 0
        self._total: float | None = None
        self._has_data = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if self._log._is_tty:
            click.echo(f"  {SPINNER[0]} {self._desc}...", nl=False)
            self._log._needs_newline = True
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def _format(self) -> str:
        spin = SPINNER[self._spin_idx % len(SPINNER)]
        if not self._has_data:
            return f"  {spin} {self._desc}..."
        current, total = self._current, self._total
        if self._unit:
            if total and total > 0:
                pct = min(100.0, current / total * 100)
                return f"  {spin} {self._desc}: {pct:5.1f}% ({current:.0f}/{total:.0f}{self._unit})"
            return f"  {spin} {self._desc}: {current:.0f}{self._unit}"
        else:
            if total and total > 0:
                return f"  {spin} {self._desc}:  {int(current)}/{int(total)}"
            return f"  {spin} {self._desc}:  {int(current)}"

    def _animate(self) -> None:
        while not self._stop.wait(0.08):
            self._spin_idx = (self._spin_idx + 1) % len(SPINNER)
            click.echo(f"\r\033[K{self._format()}", nl=False)

    def update(self, current: float, total: float | None = None) -> None:
        self._current = current
        self._total = total
        self._has_data = True
        if not self._log._is_tty:
            if total and total > 0:
                pct = int(current / total * 10)
                if pct != self._last_pct:
                    self._last_pct = pct
                    click.echo(self._format())

    def finish(self, msg: str) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
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
        self._spin_idx = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Display immediately so all labels show "waiting..." right away
        if self._log._is_tty:
            n = len(self._labels)
            for i, lbl in enumerate(self._labels):
                click.echo(
                    f"\033[K{self._format_line(lbl)}",
                    nl=(i < n - 1),
                )
            self._lines_printed = n
            self._log._needs_newline = True
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def _format_line(self, label: str) -> str:
        spin = SPINNER[self._spin_idx % len(SPINNER)]
        if label in self._state:
            current, total = self._state[label]
            if total and total > 0:
                pct = min(100.0, current / total * 100)
                if self._unit:
                    return f"  {spin} {label} {self._desc}: {pct:5.1f}% ({current:.0f}/{total:.0f}{self._unit})"
                return f"  {spin} {label} {self._desc}: {pct:5.1f}% ({current:.0f}/{total:.0f})"
            if self._unit:
                return f"  {spin} {label} {self._desc}: {current:.0f}{self._unit}"
            return f"  {spin} {label} {self._desc}: {current:.0f}"
        return f"  {spin} {label} {self._desc}..."

    def _redraw(self) -> None:
        n = len(self._labels)
        if n > 1:
            click.echo(f"\r\033[{n - 1}A", nl=False)
        else:
            click.echo("\r", nl=False)
        for i, lbl in enumerate(self._labels):
            click.echo(
                f"\033[K{self._format_line(lbl)}",
                nl=(i < n - 1),
            )

    def _animate(self) -> None:
        while not self._stop.wait(0.08):
            with self._lock:
                self._spin_idx = (self._spin_idx + 1) % len(SPINNER)
                self._redraw()

    def update(self, label: str, current: float, total: float | None = None) -> None:
        with self._lock:
            self._state[label] = (current, total)
        if not self._log._is_tty:
            if total and total > 0:
                pct = int(current / total * 10)
                if pct != self._last_pcts.get(label, -1):
                    self._last_pcts[label] = pct
                    click.echo(self._format_line(label))

    def finish(self, msg: str) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
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


class ActivityLine:
    """Handle for an in-progress activity that can be replaced with a ✓ line."""

    def __init__(self, log: PipelineLog, msg: str, detail: str = "") -> None:
        self._log = log
        self._msg = msg
        self._detail = detail
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if self._log._is_tty:
            self._spin_idx = 0
            detail_str = f"  {click.style(detail, dim=True)}" if detail else ""
            click.echo(f"  {SPINNER[0]} {msg}{detail_str}", nl=False)
            self._log._needs_newline = True
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def _animate(self) -> None:
        detail_str = f"  {click.style(self._detail, dim=True)}" if self._detail else ""
        while not self._stop.wait(0.08):
            self._spin_idx = (self._spin_idx + 1) % len(SPINNER)
            click.echo(f"\r\033[K  {SPINNER[self._spin_idx]} {self._msg}{detail_str}", nl=False)

    def _stop_spinner(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def done(self, msg: str) -> None:
        self._stop_spinner()
        detail_str = f"  {click.style(self._detail, dim=True)}" if self._detail else ""
        if self._log._is_tty:
            click.echo(f"\r\033[K  {click.style('\u2713', fg='green')} {msg}{detail_str}")
        else:
            click.echo(f"  \u2713 {msg}{detail_str}")
        self._log._needs_newline = False
        self._log._active_activity = None


class PipelineLog:
    """Centralized progress reporting for the bilbo pipeline."""

    def __init__(self) -> None:
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._needs_newline = False
        self._first_stage = True
        self._active_activity: ActivityLine | None = None
        self._pipeline_start = time.monotonic()
        self._stage_start: float | None = None
        self._stage_label: str | None = None
        self._stage_durations: list[tuple[str, float]] = []

    def _close_line(self) -> None:
        if self._active_activity is not None:
            self._active_activity._stop_spinner()
            self._active_activity = None
        if self._needs_newline:
            click.echo()
            self._needs_newline = False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def _close_stage(self) -> None:
        if self._stage_start is not None:
            elapsed = time.monotonic() - self._stage_start
            self._stage_durations.append((self._stage_label or "?", elapsed))
            self._stage_start = None
            self._stage_label = None

    def stage(self, num: int, title: str) -> None:
        self._close_line()
        self._close_stage()
        if not self._first_stage:
            click.echo()
        self._first_stage = False
        header = f"\u2500\u2500 Stage {num}: {title} "
        header = header.ljust(50, "\u2500")
        click.echo(click.style(header, bold=True))
        self._stage_start = time.monotonic()
        self._stage_label = f"Stage {num}"

    def skip(self, msg: str) -> None:
        self._close_line()
        click.echo(click.style(f"  \u00bb {msg}", dim=True))

    def info(self, msg: str) -> None:
        self._close_line()
        click.echo(f"  {msg}")

    def detail(self, msg: str) -> None:
        """Print a dimmed, indented detail line (4-space indent)."""
        self._close_line()
        click.echo(click.style(f"    {msg}", dim=True))

    def activity(self, msg: str, detail: str = "") -> ActivityLine:
        """Print an activity line with animated spinner, replaced with ✓ when done."""
        self._close_line()
        if not self._is_tty:
            detail_str = f"  {click.style(detail, dim=True)}" if detail else ""
            click.echo(f"  {msg}{detail_str}")
        line = ActivityLine(self, msg, detail)
        self._active_activity = line
        return line

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

    def summary(self) -> None:
        self._close_line()
        self._close_stage()
        total = time.monotonic() - self._pipeline_start
        parts = ", ".join(
            f"{label}: {self._format_duration(dur)}"
            for label, dur in self._stage_durations
        )
        click.echo()
        click.echo(
            f"Done in {click.style(self._format_duration(total), bold=True)}"
            f"  {click.style(f'({parts})', dim=True)}"
        )
