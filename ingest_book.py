import os
import pickle
from pypdf import PdfReader
from openai import OpenAI
import numpy as np

# ---------- CONFIG ----------
PDF_PATH = "guidelines.pdf"
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
# ----------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        text = text.replace("\u00A0", " ").strip()
        pages.append({
            "page": i + 1,
            "text": text
        })
    return pages

def embed_text(text: str):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding

def main():
    print("Reading PDF...")
    pages = extract_pages(PDF_PATH)
    print(f"Found {len(pages)} pages with text.")

    vectors = []
    metadata = []

    for i, page in enumerate(pages):
        print(f"Embedding page {page['page']} ({i+1}/{len(pages)})...")
        vec = embed_text(page["text"])
        vectors.append(vec)
        metadata.append({
            "page": page["page"],
        })

    vectors = np.array(vectors, dtype="float32")

    print("Saving embeddings and metadata...")
    np.save(EMBEDDINGS_PATH, vectors)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Done! Book is now embedded.")

if __name__ == "__main__":
    main()