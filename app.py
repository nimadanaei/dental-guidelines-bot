import os
import pickle
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from pypdf import PdfReader

# ---------- CONFIG ----------
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "metadata.pkl"
PDF_PATH = "guidelines.pdf"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"  # or another small model
TOP_K = 4
SIMILARITY_THRESHOLD = 0.4
# ----------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Load vectors + metadata
doc_vectors = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

_pdf_reader = PdfReader(PDF_PATH)

app = FastAPI(title="Dental Guidelines Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # you can later restrict this if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI at http://127.0.0.1:8000/
@app.get("/")
def serve_index():
    return FileResponse("index.html")

class Question(BaseModel):
    question: str

def embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")

def cosine_sim(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    dot = np.dot(doc_vecs, query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    query_norm = np.linalg.norm(query_vec)
    return dot / (doc_norms * query_norm + 1e-10)

def retrieve_relevant_chunks(question: str, top_k: int = TOP_K):
    q_vec = embed_text(question)
    sims = cosine_sim(q_vec, doc_vectors)
    top_indices = np.argsort(-sims)[:top_k]
    top = []
    for idx in top_indices:
        top.append({
            "index": int(idx),
            "similarity": float(sims[idx]),
            "page": metadata[idx]["page"]
        })
    return top

def get_page_text(page_num: int) -> str:
    page = _pdf_reader.pages[page_num - 1]
    text = page.extract_text() or ""
    return text.replace("\u00A0", " ").strip()

def build_sources_text(chunks: List[dict]) -> str:
    parts = []
    for ch in chunks:
        page_num = ch["page"]
        text = get_page_text(page_num)
        parts.append(f"[Page {page_num}]\n{text}\n")
    return "\n".join(parts)

SYSTEM_PROMPT = """
You are a dental guidelines assistant whose ONLY source of information
is the 'Dental Guidelines Book' excerpts provided under "Sources".

Rules:
- You must answer ONLY using the information inside "Sources".
- Do NOT use any outside medical or dental knowledge, even if you know it.
- If the answer is not clearly supported by the Sources, you MUST say:
  "This is not specified in the guidelines book I am based on."
- Do NOT guess, extrapolate, or invent information.
- Use cautious language.
- At the end of your answer, list the pages you used.
"""

@app.post("/ask")
def ask(question: Question):
    top_chunks = retrieve_relevant_chunks(question.question)

    best_sim = top_chunks[0]["similarity"] if top_chunks else 0.0
    if best_sim < SIMILARITY_THRESHOLD:
        return {
            "answer": (
                "I’m only able to answer based on the Dental Guidelines Book, "
                "and I could not find anything clearly relevant to your question."
            ),
            "used_pages": [],
            "best_similarity": best_sim
        }

    sources_text = build_sources_text(top_chunks)
    used_pages = sorted(set(ch["page"] for ch in top_chunks))

    user_content = (
        f"Question:\n{question.question}\n\n"
        f"Sources (excerpts from the Dental Guidelines Book):\n\n"
        f"{sources_text}\n\n"
        "Remember: If the answer is not clearly supported by the Sources, "
        "you must say that it is not specified."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    )

    answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "used_pages": used_pages,
        "best_similarity": best_sim
    }