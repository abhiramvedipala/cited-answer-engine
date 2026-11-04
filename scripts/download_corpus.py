"""Download the corpus, verify sha256 checksums, and lay out data/corpus/*.pdf.

Run: uv run python scripts/download_corpus.py
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
CORPUS = Path("data/corpus")

SOURCES = {
    "python-3.12-docs-pdf-a4.zip": (
        "https://docs.python.org/3.12/archives/python-3.12-docs-pdf-a4.zip",
        "0aa8f54825a192326de0e1f51bada27b27aee7c977884216426f814582dd2218",
    ),
    "nasa-se-handbook.pdf": (
        "https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf",
        "8eeb4887a4dc57a23049da7dd2ed556833cf98e214b240468d987873164ff688",
    ),
    "nist-sp-800-53r5.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "fc63bcd61715d0181dd8e85998b1e6201ae3515fc6626102101cab1841e11ec6",
    ),
    "nist-sp-800-61r2.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf",
        "b1c5ead0632dcfb979c59d8c462b094ad2465ddaf62d5c246289d6d9b4419df1",
    ),
    "nist-sp-800-207.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
        "0290d6ece24874287316f4bf430fef770aa4ec08a2227c8f2c1e5b2ff975e03d",
    ),
}

# library.pdf (2,393 pages) and c-api.pdf (363) are left out to keep the corpus
# near 2,000 pages and the eval set from being dominated by one API reference.
PY_EXCLUDE = {"library.pdf", "c-api.pdf"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = RAW / name
        if not path.exists():
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "cited-answer-engine"})
            with urllib.request.urlopen(req) as r:
                path.write_bytes(r.read())
        got = sha256(path)
        if got != expected:
            print(f"CHECKSUM MISMATCH {name}: {got} != {expected}", file=sys.stderr)
            print("The upstream file changed; re-pin it and re-run the eval.", file=sys.stderr)
            return 1
        print(f"ok {name}")

    with zipfile.ZipFile(RAW / "python-3.12-docs-pdf-a4.zip") as z:
        for member in z.namelist():
            base = Path(member).name
            if base.endswith(".pdf") and base not in PY_EXCLUDE:
                (CORPUS / f"python-{base}").write_bytes(z.read(member))
    for name in SOURCES:
        if name.endswith(".pdf"):
            (CORPUS / name).write_bytes((RAW / name).read_bytes())
    print(f"{len(list(CORPUS.glob('*.pdf')))} PDFs in {CORPUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
