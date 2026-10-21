import hashlib
import re


def split_text(text: str, size: int = 900, overlap: int = 120) -> list:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 30]


def chunk_ids(ns: str, chunks: list) -> list:
    # same text -> same id, so re-uploading a file never creates duplicates
    return [f"{ns}-{hashlib.sha1(c.encode()).hexdigest()[:20]}" for c in chunks]