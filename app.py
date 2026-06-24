"""
FastAPI server for the Oral & Dental Guidelines bot.

Run locally:
    export OPENAI_API_KEY=sk-...
    uvicorn app:app --reload

Endpoints:
    GET  /            -> web UI (index.html)
    GET  /health      -> readiness/liveness check
    POST /ask         -> {answer, used_pages, citations, best_similarity}
    POST /ask/stream  -> server-sent token stream (nicer UX)
"""
import json
import pickle
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core import (
    EMBEDDINGS_PATH, METADATA_PATH,
    CHAT_MODEL, TOP_K, CANDIDATE_K, SIMILARITY_THRESHOLD, MAX_QUESTION_CHARS,
    GUIDELINE_TITLE, GUIDELINE_VERSION,
    get_client, embed_one, l2_normalize, rerank,
    LexicalIndex, hybrid_scores, adult_dose_flags,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dental-bot")

# ---------------------------------------------------------------------------
# Load index once at startup
# ---------------------------------------------------------------------------
doc_vectors = np.load(EMBEDDINGS_PATH)          # already L2-normalised by ingest
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)
assert len(metadata) == doc_vectors.shape[0], "index/metadata length mismatch"
logger.info("Loaded %d chunks, dim=%d", *doc_vectors.shape)

# Lexical (BM25) index over chunk text, for hybrid retrieval
lexical = LexicalIndex([m["text"] for m in metadata])
adult_flags = adult_dose_flags([m["text"] for m in metadata])
logger.info("Built lexical index over %d chunks (%d adult-dosing chunks)",
            lexical.N, int(adult_flags.sum()))

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Oral & Dental Guidelines Bot")
app.state.limiter = limiter

# Frontend is served from this same app, so CORS can stay closed by default.
# Add specific origins here only if you embed the widget on another domain.
ALLOWED_ORIGINS: List[str] = []  # e.g. ["https://your-site.edu.au"]
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )


@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"answer": "You're sending questions too quickly. "
                           "Please wait a moment and try again.",
                 "used_pages": [], "citations": [], "best_similarity": 0.0},
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class Turn(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class Question(BaseModel):
    question: str
    history: List[Turn] = []   # recent prior turns, oldest first

    @field_validator("question")
    @classmethod
    def _validate(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Question must not be empty.")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question exceeds {MAX_QUESTION_CHARS} characters.")
        return v

    @field_validator("history")
    @classmethod
    def _cap_history(cls, v):
        # keep only the last 6 turns, and clamp each to a sane length
        v = v[-6:]
        for t in v:
            t.content = (t.content or "")[:2000]
        return v


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(question: str, top_k: int = TOP_K):
    # vector similarity
    q_vec = l2_normalize(embed_one(question)[None, :])[0]
    vec_sims = doc_vectors @ q_vec                  # cosine (both normalised)

    # lexical similarity (BM25) — guarantees term matches get weight
    lex = lexical.scores(question)

    # fuse, then take the top candidates by the fused score
    fused = hybrid_scores(vec_sims, lex, adult_flags=adult_flags, query=question)
    pool_idx = np.argsort(-fused)[:CANDIDATE_K]

    candidates = [
        {
            "chunk_id": int(i),
            "page": int(metadata[i]["page"]),
            "text": metadata[i]["text"],
            "similarity": float(vec_sims[i]),   # kept for the gate + display
            "fused": float(fused[i]),
            "vec": doc_vectors[i],
        }
        for i in pool_idx
    ]

    ranked = rerank(candidates, top_k)

    out = []
    for c in ranked:
        c = {k: v for k, v in c.items()
             if not k.startswith("_") and k not in ("vec", "fused")}
        out.append(c)
    return out


def build_sources_block(chunks: List[dict]) -> str:
    return "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)


SYSTEM_PROMPT = f"""You are a careful assistant whose ONLY source of information is \
the excerpts from "{GUIDELINE_TITLE}, {GUIDELINE_VERSION}" provided under "Sources".

Core rules:
- Answer ONLY using information inside "Sources". Do not use outside medical or \
dental knowledge, even if you are confident in it.
- If the answer is not clearly and explicitly supported by the Sources, reply \
exactly: "This is not specified in the guideline excerpts I can access."
- Never guess, extrapolate, or invent drugs, doses, frequencies, or durations.

Giving doses and regimens:
- Reproduce doses, frequencies, and durations VERBATIM from the Sources (e.g. \
"ibuprofen 400 mg orally, 6- to 8-hourly"). Do not round, summarise, or convert numbers.
- Give the COMPLETE regimen, not a fragment. If the Sources pair drugs (e.g. an \
NSAID PLUS paracetamol), state every component with its own dose, not just one.
- When a question doesn't specify the patient's age, and the Sources contain BOTH \
adult and child/paediatric dosing, lead with the ADULT regimen in full, then add a \
short line noting that weight-based paediatric dosing is also available and asking \
whether the patient is a child. Do not give only the paediatric/liquid dosing to a \
question that didn't ask for it.
- If the Sources show a drug as part of a stepwise or combination approach, present \
that structure (first-line, then additions) rather than a single isolated line.

Style:
- Cite the page inline after a dose or claim, e.g. "(p. 142)".
- Use cautious, clinical language. This supports — never replaces — a clinician's \
own judgement and the current published guideline.
- End with a line: "Pages used: <comma-separated page numbers>"."""


def rewrite_query(question: str, history) -> str:
    """Turn a short follow-up ("adult", "what about children?") into a
    standalone retrieval query using the recent conversation. Long, specific
    questions are returned unchanged to avoid wasting a model call."""
    if not history:
        return question
    # Heuristic: only rewrite when the new question is short / context-dependent.
    words = question.split()
    looks_followup = (len(words) <= 6) or question.lower().lstrip().startswith(
        ("and ", "what about", "for ", "in ", "adult", "child", "how about",
         "that", "it", "this", "those", "them", "yes", "no"))
    if not looks_followup:
        return question

    convo = "\n".join(f"{t.role}: {t.content}" for t in history[-4:])
    try:
        resp = get_client().chat.completions.create(
            model=CHAT_MODEL, temperature=0, max_tokens=80,
            messages=[
                {"role": "system", "content":
                    "Rewrite the user's latest message into a single standalone "
                    "search query for a dental guidelines reference, resolving any "
                    "pronouns or context from the conversation. Output ONLY the "
                    "query, no preamble."},
                {"role": "user", "content":
                    f"Conversation:\n{convo}\n\nLatest message: {question}\n\n"
                    "Standalone query:"},
            ],
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception:
        logger.exception("query rewrite failed; using raw question")
        return question


def build_messages(question: str, chunks: List[dict], history=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # include prior turns so the model has conversational context
    for t in (history or []):
        role = "assistant" if t.role == "assistant" else "user"
        msgs.append({"role": role, "content": t.content})
    user = (
        f"Question:\n{question}\n\n"
        f"Sources (excerpts from {GUIDELINE_TITLE}, {GUIDELINE_VERSION}):\n\n"
        f"{build_sources_block(chunks)}\n\n"
        "If the answer is not clearly supported above, say it is not specified. "
        "Use the conversation above to resolve what the question refers to."
    )
    msgs.append({"role": "user", "content": user})
    return msgs


def below_gate_response(best_sim: float):
    return {
        "answer": ("I can only answer from "
                   f"{GUIDELINE_TITLE}, {GUIDELINE_VERSION}, and I couldn't find a "
                   "passage clearly relevant to your question. Try rephrasing with "
                   "the specific drug, condition, or procedure."),
        "used_pages": [],
        "citations": [],
        "best_similarity": best_sim,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": int(doc_vectors.shape[0]),
            "model": CHAT_MODEL, "version": GUIDELINE_VERSION}


@app.post("/ask")
@limiter.limit("20/minute")
def ask(request: Request, payload: Question):
    search_query = rewrite_query(payload.question, payload.history)
    chunks = retrieve(search_query)
    best_sim = max((c["similarity"] for c in chunks), default=0.0)
    if best_sim < SIMILARITY_THRESHOLD:
        return below_gate_response(best_sim)

    try:
        resp = get_client().chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            max_tokens=700,
            messages=build_messages(payload.question, chunks, payload.history),
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        logger.exception("chat completion failed")
        raise HTTPException(status_code=502,
                            detail="The language model is temporarily unavailable.")

    used_pages = sorted({c["page"] for c in chunks})
    citations = [{"page": c["page"], "similarity": round(c["similarity"], 3),
                  "preview": c["text"][:240]} for c in chunks]
    return {"answer": answer, "used_pages": used_pages,
            "citations": citations, "best_similarity": best_sim}


@app.post("/ask/stream")
@limiter.limit("20/minute")
def ask_stream(request: Request, payload: Question):
    search_query = rewrite_query(payload.question, payload.history)
    chunks = retrieve(search_query)
    best_sim = max((c["similarity"] for c in chunks), default=0.0)

    def gen():
        if best_sim < SIMILARITY_THRESHOLD:
            yield _sse({"type": "meta", "used_pages": [], "citations": [],
                        "best_similarity": best_sim})
            yield _sse({"type": "token",
                        "text": below_gate_response(best_sim)["answer"]})
            yield _sse({"type": "done"})
            return

        used_pages = sorted({c["page"] for c in chunks})
        citations = [{"page": c["page"], "similarity": round(c["similarity"], 3),
                      "preview": c["text"][:240]} for c in chunks]
        yield _sse({"type": "meta", "used_pages": used_pages,
                    "citations": citations, "best_similarity": best_sim})
        try:
            stream = get_client().chat.completions.create(
                model=CHAT_MODEL, temperature=0, max_tokens=700,
                messages=build_messages(payload.question, chunks, payload.history),
                stream=True,
            )
            for part in stream:
                delta = part.choices[0].delta.content
                if delta:
                    yield _sse({"type": "token", "text": delta})
        except Exception:
            logger.exception("stream failed")
            yield _sse({"type": "token",
                        "text": "\n\n[The language model became unavailable.]"})
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
