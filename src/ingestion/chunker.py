"""Token-based sliding-window chunker with section preservation."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from src.ingestion.pdf_parser import TextBlock


@dataclass
class TextChunk:
    """A text chunk with section metadata."""

    text: str
    section: str | None
    chunk_index: int


def _encode_tokens(text: str, enc: tiktoken.Encoding) -> list[int]:
    return enc.encode(text)


def _decode_tokens(tokens: list[int], enc: tiktoken.Encoding) -> str:
    return enc.decode(tokens)


def chunk_blocks(
    blocks: list[TextBlock],
    *,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[TextChunk]:
    """Split blocks into overlapping token windows; keep dominant section per chunk."""
    if not blocks:
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    # Flatten to (token_ids, section) segments
    segments: list[tuple[list[int], str | None]] = []
    for block in blocks:
        tokens = _encode_tokens(block.text, enc)
        if tokens:
            segments.append((tokens, block.section))

    if not segments:
        return []

    flat_tokens: list[int] = []
    token_sections: list[str | None] = []
    for tokens, section in segments:
        flat_tokens.extend(tokens)
        token_sections.extend([section] * len(tokens))

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    step = max(chunk_size - overlap, 1)

    while start < len(flat_tokens):
        end = min(start + chunk_size, len(flat_tokens))
        chunk_tokens = flat_tokens[start:end]
        if not chunk_tokens:
            break

        section_counts: dict[str | None, int] = {}
        for sec in token_sections[start:end]:
            section_counts[sec] = section_counts.get(sec, 0) + 1
        dominant_section = max(section_counts, key=lambda k: section_counts[k])

        chunks.append(
            TextChunk(
                text=_decode_tokens(chunk_tokens, enc).strip(),
                section=dominant_section,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1
        if end >= len(flat_tokens):
            break
        start += step

    return chunks
