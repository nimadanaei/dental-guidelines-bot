"""
Ingest the guidelines PDF into a chunked vector index.

Run once (or whenever the PDF changes):

    export OPENAI_API_KEY=sk-...
    python ingest_book.py

Produces:
    embeddings.npy   (n_chunks, EMBEDDING_DIM) float32, L2-normalised
    metadata.pkl     list of {page, text, chunk_id}

Optional, to help you pick SIMILARITY_THRESHOLD from real data:
    python ingest_book.py --tune
"""
import sys
import pickle
import time

import numpy as np
from pypdf import PdfReader

from core import (
    PDF_PATH, EMBEDDINGS_PATH, METADATA_PATH,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    Chunk, clean_text, chunk_page, embed_texts, embed_one, l2_normalize,
)

EMBED_BATCH = 64  # how many chunks per OpenAI embedding call


def build_chunks() -> list[Chunk]:
    reader = PdfReader(PDF_PATH)
    print(f"Reading {len(reader.pages)} pages from {PDF_PATH} ...")
    chunks: list[Chunk] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        for ch in chunk_page(raw, page=i + 1):
            ch.chunk_id = len(chunks)
            chunks.append(ch)
    print(f"Produced {len(chunks)} chunks "
          f"(avg {sum(len(c.text) for c in chunks)//max(len(chunks),1)} chars).")
    return chunks


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    vectors = np.zeros((len(chunks), EMBEDDING_DIM), dtype="float32")
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        for attempt in range(3):
            try:
                vecs = embed_texts([c.text for c in batch])
                vectors[start:start + len(batch)] = vecs
                break
            except Exception as e:  # transient API/network errors
                wait = 2 ** attempt
                print(f"  batch {start} failed ({e}); retry in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding batch starting at {start} failed 3x.")
        print(f"  embedded {min(start + EMBED_BATCH, len(chunks))}/{len(chunks)}")
    return l2_normalize(vectors)


def main():
    chunks = build_chunks()
    vectors = embed_chunks(chunks)
    np.save(EMBEDDINGS_PATH, vectors)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(
            [{"page": c.page, "text": c.text, "chunk_id": c.chunk_id} for c in chunks],
            f,
        )
    print(f"Saved {EMBEDDINGS_PATH} {vectors.shape} and {METADATA_PATH}.")
    print(f"Embedding model: {EMBEDDING_MODEL} @ {EMBEDDING_DIM} dims. Done.")


def tune():
    """Print best-chunk similarity for a set of real questions so you can set
    SIMILARITY_THRESHOLD from evidence instead of guessing."""
    vectors = np.load(EMBEDDINGS_PATH)
    probes = [
        "first-line analgesics for mild to moderate postoperative dental pain",
        "antibiotic regimen for acute odontogenic infection",
        "when should ibuprofen be avoided or used with caution",
        "infective endocarditis prophylaxis before dental procedures",
        "management of dry socket alveolar osteitis",
        # deliberately off-topic — best score here should fall BELOW your gate
        "how do I change a car tyre",
        "what is the capital of France",
    ]
    print(f"{'best_sim':>9}  question")
    for q in probes:
        qv = l2_normalize(embed_one(q)[None, :])[0]
        sims = vectors @ qv
        print(f"{float(np.max(sims)):>9.3f}  {q}")
    print("\nSet SIMILARITY_THRESHOLD between the lowest on-topic score and the "
          "highest off-topic score.")


if __name__ == "__main__":
    if "--tune" in sys.argv:
        tune()
    else:
        main()
