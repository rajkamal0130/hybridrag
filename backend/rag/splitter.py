"""Split: recursive character chunks with overlap. Each chunk gets a stable chunk_id."""
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    out, buf = [], ""
    for piece in text.split(sep):
        candidate = f"{buf}{sep}{piece}" if buf else piece
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(piece) > size:
            out.extend(_recursive_split(piece, size, rest))
            buf = ""
        else:
            buf = piece
    if buf:
        out.append(buf)
    return out


def split_segments(segments: list[dict]) -> list[dict]:
    chunks = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        parts = _recursive_split(text, CHUNK_SIZE - CHUNK_OVERLAP, SEPARATORS)
        for i, part in enumerate(parts):
            if i > 0 and CHUNK_OVERLAP > 0:
                part = parts[i - 1][-CHUNK_OVERLAP:] + part
            part = part.strip()
            if part:
                chunks.append({
                    "chunk_id": f"{seg['source']}::p{seg['page']}::c{i}",
                    "text": part,
                    "source": seg["source"],
                    "page": seg["page"],
                })
    return chunks
