"""PDF -> one text record per page. Page numbers are 1-based, as printed by PDF viewers."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Page:
    doc: str
    page: int
    text: str


def parse_pdf(path: Path) -> list[Page]:
    with pymupdf.open(path) as pdf:
        # NUL bytes appear in some PDF text layers and Postgres text rejects them
        return [Page(path.stem, i + 1, p.get_text().replace("\x00", "")) for i, p in enumerate(pdf)]


def parse_dir(corpus: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(corpus.glob("*.pdf")):
        pages.extend(parse_pdf(path))
    return pages
