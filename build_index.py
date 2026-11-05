"""Build the vector index used by bank_agent.search.

Pipeline: load the PDF reports -> extract text page by page -> split pages into
overlapping chunks -> embed the chunks with Gemini -> save
data/embeddings.npy (one row per chunk) and data/chunks.json (chunk metadata,
in the same order as the matrix rows).

Run once, or whenever the source PDFs change:  python build_index.py
"""

import json

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

PDF_FILES = [
    "data/Annual-Report-2025-eng.pdf",
    "data/audit-report-2025-eng-1777360685-.pdf",
]
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 20  # number of chunks sent per embedding request


def extract_pages(paths: list[str]) -> list[dict]:
    """Return one {"source", "page", "text"} dict per PDF page (pages numbered from 1)."""
    pages = []
    for path in paths:
        reader = PdfReader(path)
        for number, page in enumerate(reader.pages, start=1):
            pages.append({"source": path, "page": number, "text": page.extract_text()})
    return pages


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into pieces of `size` characters.

    Each piece repeats the last `overlap` characters of the previous one, so a
    sentence that straddles a boundary is still fully contained in one chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split every page into chunks, keeping the source file and page number for citations."""
    chunks = []
    for p in pages:
        # Scanned or blank pages yield empty text; they would only add noise to the index.
        if not p["text"].strip():
            continue
        for piece in chunk_text(p["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"source": p["source"], "page": p["page"], "text": piece})
    return chunks


def embed_chunks(client: genai.Client, chunks: list[dict]) -> np.ndarray:
    """Embed all chunks in batches and return a (num_chunks, dims) matrix."""
    all_vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[c["text"] for c in batch],
            # Documents and queries are embedded with matching task types
            # (RETRIEVAL_DOCUMENT here, RETRIEVAL_QUERY at search time).
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_vectors.extend(np.array(e.values) for e in result.embeddings)
        print(f"embedded {i + len(batch)} / {len(chunks)}")
    return np.array(all_vectors)


def main() -> None:
    """Build the index and write it to data/."""
    load_dotenv()  # credentials come from the environment / .env, never from code
    client = genai.Client()

    pages = extract_pages(PDF_FILES)
    chunks = chunk_pages(pages)
    matrix = embed_chunks(client, chunks)
    print("Matrix shape:", matrix.shape)

    np.save("data/embeddings.npy", matrix)
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f)
    print("Saved data/embeddings.npy and data/chunks.json")


if __name__ == "__main__":
    main()
