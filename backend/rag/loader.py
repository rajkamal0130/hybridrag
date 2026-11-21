"""Load: read a PDF into one segment per page."""
from pathlib import Path

from pypdf import PdfReader


def load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    return [
        {"text": page.extract_text() or "", "source": path.name, "page": i + 1}
        for i, page in enumerate(reader.pages)
    ]
