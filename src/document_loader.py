"""Load and extract plain text from various file formats."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class Document:
    content: str
    source: str    # absolute path (used as stable ID)
    filename: str  # basename only (shown to users in sources)
    file_type: str # extension without leading dot
    mtime: float   # st_mtime for change detection
    size: int      # st_size for change detection


def scan_directory(dataset_path: str, supported_extensions: list) -> list:
    """Recursively collect all supported files under dataset_path."""
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    ext_set = {e.lower() for e in supported_extensions}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    return sorted(files)


def load_document(path: Path) -> Optional[Document]:
    """Extract text from a single file. Returns None if unreadable or empty."""
    ext = path.suffix.lower()
    try:
        if ext in (".html", ".htm"):
            content = _load_html(path)
        elif ext == ".pdf":
            content = _load_pdf(path)
        elif ext == ".docx":
            content = _load_docx(path)
        elif ext == ".json":
            content = _load_json(path)
        else:
            # Plain text family: md, txt, cls, trigger, js, ts, xml, soql …
            content = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        print(f"  [WARN] Could not load {path.name}: {exc}")
        return None

    if not content or len(content.strip()) < 30:
        return None

    stat = path.stat()
    return Document(
        content=content.strip(),
        source=str(path.resolve()),
        filename=path.name,
        file_type=ext.lstrip("."),
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    # Strip chrome elements that add noise
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "noscript", "aside", "iframe"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Prefer <main>/<article> for cleaner content; fall back to <body>
    body = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text = body.get_text(separator="\n", strip=True)

    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Prepend title so every chunk can mention the article name
    if title and title not in text[:300]:
        text = f"{title}\n\n{text}"

    return text


def _load_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required for PDF support: pip install pypdf")
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _load_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDoc
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")
    doc = DocxDoc(str(path))
    return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def _load_json(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw
