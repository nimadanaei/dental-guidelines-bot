"""
Shared configuration and core utilities for the Oral & Dental Guidelines bot.

Both ingest_book.py and app.py import from here so the embedding model,
chunking logic, and vector maths never drift apart between the two.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG  (single source of truth for both ingest and serve)
# ---------------------------------------------------------------------------
PDF_PATH = "guidelines.pdf"
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "metadata.pkl"

# NOTE: the original embeddings.npy was 1536-dim, which is text-embedding-3-large
# at reduced dimensions OR ada-002. We standardise on 3-large here. If you change
# this you MUST re-run ingest_book.py, or query and document vectors won't match.
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1536          # request reduced dims to keep files small + fast
CHAT_MODEL = "gpt-4.1-mini"

# Chunking
CHUNK_TARGET_CHARS = 1400     # ~350 tokens, a coherent passage not a whole page
CHUNK_OVERLAP_CHARS = 200     # carry context across boundaries

# Retrieval
TOP_K = 8               # how many chunks the model finally sees (was 5)
CANDIDATE_K = 24        # how many to pull before re-ranking, then trim to TOP_K
# Gate is applied to the BEST chunk. With 3-large, relevant dental passages
# typically score ~0.30-0.55. Tune with tune_threshold() in ingest_book.py.
SIMILARITY_THRESHOLD = 0.30

# Versioning — surfaced in the UI and answers so the label can never lie
GUIDELINE_TITLE = "Therapeutic Guidelines: Oral and Dental"
GUIDELINE_VERSION = "Version 4 (2025 update)"

# Limits
MAX_QUESTION_CHARS = 1000

# ---------------------------------------------------------------------------
# OpenAI client (lazy so importing this module never crashes without a key)
# ---------------------------------------------------------------------------
_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        _client = OpenAI(api_key=key)
    return _client


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    page: int
    text: str
    chunk_id: int = field(default=-1)


# ---------------------------------------------------------------------------
# Text cleaning + chunking
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00A0", " ")
    # collapse runs of whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(text: str, page: int,
               target: int = CHUNK_TARGET_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> List[Chunk]:
    """Split one page's text into overlapping passages on sentence-ish
    boundaries. Keeps chunks coherent so each embedding means one thing."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= target:
        return [Chunk(page=page, text=text)]

    # split on sentence / line boundaries, then greedily pack to target size
    pieces = re.split(r"(?<=[.;:])\s+|\n+", text)
    chunks: List[Chunk] = []
    buf = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(buf) + len(piece) + 1 <= target:
            buf = f"{buf} {piece}".strip()
        else:
            if buf:
                chunks.append(Chunk(page=page, text=buf))
            # start new buffer carrying the tail of the previous one as overlap
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail} {piece}".strip()
    if buf:
        chunks.append(Chunk(page=page, text=buf))
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + similarity
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Batch-embed a list of strings. Returns an (n, dim) float32 array."""
    resp = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
    )
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")


def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / (norms + 1e-10)


# ---------------------------------------------------------------------------
# Hybrid retrieval: lexical (BM25-style) + vector, fused.
# ---------------------------------------------------------------------------
# Why: vector search alone is weak on short keyword queries ("ibuprofen dose").
# A bare term can embed closer to a dense repetitive table than to the one clean
# adult-dosing sentence. Adding a lexical score guarantees that chunks literally
# containing the queried terms get real weight, which reliably surfaces the right
# passage. We then de-weight table dumps and penalise near-duplicates.

import re as _re
import math as _math

_TABLE_HINTS = ("mg/kg", "ml liquid", "weight:", "volume of", "see table")
_TOKEN = _re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


class LexicalIndex:
    """Tiny BM25 over the chunk corpus. Built once at startup."""
    def __init__(self, texts: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(t) for t in texts]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = []
        df = {}
        for d in self.docs:
            counts = {}
            for w in d:
                counts[w] = counts.get(w, 0) + 1
            self.tf.append(counts)
            for w in counts:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: _math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for w, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = [w for w in _tokenize(query) if w in self.idf]
        out = np.zeros(self.N, dtype="float32")
        if not q:
            return out
        for i, counts in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for w in q:
                f = counts.get(w, 0)
                if f:
                    idf = self.idf[w]
                    s += idf * (f * (self.k1 + 1)) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _looks_like_table(text: str) -> float:
    """0..1 score for how 'table-dump' a chunk looks (more = denser table)."""
    t = text.lower()
    hits = sum(t.count(h) for h in _TABLE_HINTS)
    digits = sum(c.isdigit() for c in t)
    ratio = digits / max(len(t), 1)
    return min(1.0, hits / 6.0) * 0.6 + min(1.0, ratio / 0.12) * 0.4


# A clean adult ORAL dosing line: "<drug> NNN mg orally", in adult context,
# without paediatric weight-table markers. This is the signal that reliably
# distinguishes the one adult-regimen sentence from a wall of paediatric rows.
_DOSE_PAT = _re.compile(
    r"\b[a-z]+\s+\d+\s*(?:mg|g|microgram|mcg)\s+orally", _re.I)
_ADULT_PAT = _re.compile(r"\badults?\b", _re.I)
_PAED_MARK = ("mg/kg", "ml liquid", "weight:")


def adult_dose_flags(texts: list) -> np.ndarray:
    """1.0 for chunks that are clean adult oral-dosing prose, else 0.0.
    Looks at each chunk plus its predecessor (section headings like
    '...regimens for ... pain in adults' often sit in the prior chunk)."""
    flags = np.zeros(len(texts), dtype="float32")
    for i, t in enumerate(texts):
        tl = t.lower()
        if not _DOSE_PAT.search(t):
            continue
        if any(m in tl for m in _PAED_MARK):
            continue
        ctx = t + " " + (texts[i - 1] if i > 0 else "")
        if _ADULT_PAT.search(ctx):
            flags[i] = 1.0
    return flags


def hybrid_scores(vec_sims: np.ndarray, lex_scores: np.ndarray,
                  adult_flags: np.ndarray = None, query: str = "",
                  w_vec: float = 0.5, w_lex: float = 0.4) -> np.ndarray:
    """Fuse normalised vector + lexical scores, then add a targeted boost for
    clean adult-dosing chunks when the query is about dosing/pain/medication."""
    fused = w_vec * _minmax(vec_sims) + w_lex * _minmax(lex_scores)
    if adult_flags is not None:
        ql = query.lower()
        dose_query = any(w in ql for w in (
            "dos", "mg", "regimen", "analges", "pain", "ibuprofen", "paracetamol",
            "nsaid", "celecoxib", "amoxicillin", "metronidazole", "antibiotic",
            "prescri", "how much", "adult"))
        if dose_query:
            fused = fused + 0.30 * adult_flags
    return fused


def rerank(candidates: list, top_k: int) -> list:
    """Greedy selection over candidates (each has 'fused','text','vec').
    De-weights dense table dumps and penalises near-duplicate chunks so a clean
    prose dose line surfaces alongside any tables."""
    if not candidates:
        return []
    pool = []
    for c in candidates:
        c = dict(c)
        c["_table"] = _looks_like_table(c["text"])
        c["_base"] = c.get("fused", c.get("similarity", 0.0)) - 0.05 * c["_table"]
        pool.append(c)

    selected = []
    while pool and len(selected) < top_k:
        best, best_score = None, -1e9
        for c in pool:
            redundancy = 0.0
            for s in selected:
                if "vec" in c and "vec" in s:
                    redundancy = max(redundancy, float(np.dot(c["vec"], s["vec"])))
            score = c["_base"] - 0.25 * redundancy
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        pool.remove(best)
    return selected
