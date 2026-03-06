from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .models import BookMeta

DEFAULT_LIBRARY = Path.home() / ".bilbo"


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


class Library:
    def __init__(self, root: Path | None = None):
        self.root = root or DEFAULT_LIBRARY
        self.books_dir = self.root / "books"
        self.index_path = self.root / "library.json"

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.books_dir.mkdir(exist_ok=True)
        if not self.index_path.exists():
            self._write_index({})

    def _read_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text())

    def _write_index(self, index: dict[str, dict]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2))
        tmp.rename(self.index_path)

    def list_books(self) -> list[BookMeta]:
        index = self._read_index()
        return [BookMeta.from_dict(v) for v in index.values()]

    def get(self, slug: str) -> BookMeta | None:
        index = self._read_index()
        if slug in index:
            return BookMeta.from_dict(index[slug])
        return None

    def book_dir(self, slug: str) -> Path:
        return self.books_dir / slug

    def add_or_update(self, meta: BookMeta) -> None:
        self.init()
        d = self.book_dir(meta.slug)
        d.mkdir(parents=True, exist_ok=True)
        (d / "exports").mkdir(exist_ok=True)
        index = self._read_index()
        index[meta.slug] = meta.to_dict()
        self._write_index(index)

    def delete(self, slug: str) -> bool:
        index = self._read_index()
        if slug not in index:
            return False
        del index[slug]
        self._write_index(index)
        d = self.book_dir(slug)
        if d.exists():
            shutil.rmtree(d)
        return True

    def find_slug(self, title: str) -> str | None:
        """Return existing slug for this title, or None."""
        slug = _slugify(title) or "book"
        if slug in self._read_index():
            return slug
        return None

    def make_slug(self, title: str) -> str:
        slug = _slugify(title)
        if not slug:
            slug = "book"
        existing = self._read_index()
        if slug not in existing:
            return slug
        i = 2
        while f"{slug}-{i}" in existing:
            i += 1
        return f"{slug}-{i}"
