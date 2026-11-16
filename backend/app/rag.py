"""Knowledge boxes: per-user document stores with per-agent access control ("selective RAG").

The access check lives HERE, in the retrieval query, not in the prompt. An agent that is not
assigned a box cannot retrieve from it no matter what it (or a prompt injection) asks for.
"""
import hashlib
import io
import os
import re

import numpy as np
from sqlmodel import select

from . import llm, telemetry
from .config import settings
from .db import AgentBox, Box, Chunk, Document, session

LOCAL_DIM = 512


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="replace")


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunks of roughly `size` characters with a little overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # a single huge paragraph: hard-split it
            head, p = p[:size], p[size - overlap:]
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(head)
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _local_embed(text: str) -> np.ndarray:
    """The "local" embedding option: feature-hashed words + bigrams. Lexical, not semantic, but needs no key."""
    vec = np.zeros(LOCAL_DIM, dtype=np.float32)
    toks = re.findall(r"[a-z0-9]+", text.lower())
    for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        vec[h % LOCAL_DIM] += 1.0 if (h >> 63) else -1.0
    return vec


async def embed(key: str, texts: list[str]) -> np.ndarray:
    if settings.embedding_model == "local":
        mat = np.stack([_local_embed(t) for t in texts])
    else:
        mat = np.array(await llm.embed(key, texts), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1, norms)


async def ingest(key: str, user_id: str, box_id: str, filename: str, data: bytes) -> Document:
    chunks = chunk_text(extract_text(filename, data))
    if not chunks:
        raise ValueError("No text could be extracted from this file.")
    vectors = await embed(key, chunks)

    safe = os.path.basename(filename)
    folder = os.path.join(settings.data_dir, "uploads", user_id, box_id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, safe), "wb") as f:  # keep the original so boxes can be re-indexed
        f.write(data)

    with session() as s:
        doc = Document(user_id=user_id, box_id=box_id, filename=safe, n_chunks=len(chunks))
        s.add(doc)
        for text, vec in zip(chunks, vectors):
            s.add(Chunk(box_id=box_id, document_id=doc.id, text=text, embedding=vec.tobytes()))
        s.commit()
        return doc


def allowed_boxes(agent_id: str) -> list[Box]:
    with session() as s:
        ids = s.exec(select(AgentBox.box_id).where(AgentBox.agent_id == agent_id)).all()
        return list(s.exec(select(Box).where(Box.id.in_(ids))).all()) if ids else []


async def search(key: str, agent_id: str, query: str, k: int = 5) -> list[dict]:
    """Top-k chunks across the boxes this agent is assigned. Unassigned boxes are never read."""
    boxes = {b.id: b.name for b in allowed_boxes(agent_id)}
    with telemetry.observation("knowledge-search", as_type="retriever",
                               input={"query": query, "boxes": sorted(boxes.values())}) as obs:
        if not boxes:
            obs.update(output=[])
            return []
        with session() as s:
            chunks = s.exec(select(Chunk).where(Chunk.box_id.in_(list(boxes)))).all()
            docs = {d.id: d.filename for d in s.exec(select(Document).where(Document.box_id.in_(list(boxes)))).all()}
        q = (await embed(key, [query]))[0]
        hits = []
        for c in chunks:
            vec = np.frombuffer(c.embedding, dtype=np.float32)
            if vec.shape == q.shape:  # skip chunks embedded with a different model
                hits.append((float(vec @ q), c))
        hits.sort(key=lambda h: h[0], reverse=True)
        out = [{"score": round(sc, 3), "box": boxes[c.box_id], "document": docs.get(c.document_id, "?"),
                "text": c.text} for sc, c in hits[:k]]
        obs.update(output=[{k_: v for k_, v in h.items() if k_ != "text"} for h in out])
        return out
