from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .models import BookMeta

DEFAULT_LIBRARY = Path.home() / ".bilbo"


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "book"


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

    def make_slug(self, title: str) -> str:
        base = _slugify(title)
        existing = self._read_index()
        if base not in existing:
            return base
        i = 2
        while f"{base}-{i}" in existing:
            i += 1
        return f"{base}-{i}"

    def find_by_title(self, title: str) -> BookMeta | None:
        index = self._read_index()
        for v in index.values():
            meta = BookMeta.from_dict(v)
            if meta.title == title:
                return meta
        return None

    def rename(self, old_title: str, new_title: str) -> BookMeta | None:
        """Rename a book: update title, slug, and move data directory."""
        meta = self.find_by_title(old_title)
        if meta is None:
            return None
        old_slug = meta.slug
        new_slug = self.make_slug(new_title)
        old_dir = self.book_dir(old_slug)
        new_dir = self.book_dir(new_slug)
        if old_dir.exists():
            old_dir.rename(new_dir)
        meta.l1_audio = meta.l1_audio.replace(old_slug, new_slug)
        meta.l2_audio = meta.l2_audio.replace(old_slug, new_slug)
        meta.title = new_title
        meta.slug = new_slug
        index = self._read_index()
        del index[old_slug]
        index[new_slug] = meta.to_dict()
        self._write_index(index)
        return meta
