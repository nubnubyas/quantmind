"""PDF text extraction with heuristic section detection (PyMuPDF)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class TextBlock:
    """A contiguous text block from a PDF page."""

    text: str
    section: str | None
    page_num: int


_SECTION_RE = re.compile(
    r"^(\d+\.?\s+)?(introduction|abstract|conclusion|references|"
    r"methodology|methods|results|discussion|background|related work|"
    r"literature review|data|empirical analysis|model|appendix)\b",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z][\w\s\-]{2,60}$")


def _is_heading(text: str, font_size: float, body_size: float) -> bool:
    """Heuristic: larger font, numbered title, or known section keyword."""
    t = text.strip()
    if len(t) < 3 or len(t) > 120:
        return False
    if font_size > body_size * 1.15:
        return True
    if _SECTION_RE.match(t):
        return True
    if _NUMBERED_HEADING_RE.match(t):
        return True
    if t.isupper() and len(t.split()) <= 8:
        return True
    return False


def parse_pdf(pdf_path: Path) -> list[TextBlock]:
    """Parse PDF into text blocks with section labels."""
    doc = fitz.open(pdf_path)
    blocks: list[TextBlock] = []
    current_section: str | None = None

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        sizes = [
            span["size"]
            for block in page_dict.get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        body_size = sum(sizes) / len(sizes) if sizes else 10.0

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            max_font = body_size
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    max_font = max(max_font, span.get("size", body_size))
                    lines.append(span.get("text", ""))
            text = " ".join(lines).strip()
            if not text:
                continue

            if _is_heading(text, max_font, body_size):
                current_section = text
                continue

            blocks.append(
                TextBlock(text=text, section=current_section, page_num=page_num)
            )

    doc.close()
    return blocks
