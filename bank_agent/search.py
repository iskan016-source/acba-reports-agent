"""Semantic search over the pre-built index of Acba Bank's 2025 reports."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# Short, consistent source labels so every agent cites documents the same way.
SOURCE_NAMES = {
    "data/Annual-Report-2025-eng.pdf": "Annual Report 2025",
    "data/audit-report-2025-eng-1777360685-.pdf": "Audited FS 2025",
}

# The index is built offline by build_index.py and loaded once at import, so a
# search costs only one embedding call for the question.
DATA = Path(__file__).parent.parent / "data"

matrix = np.load(DATA / "embeddings.npy")
with open(DATA / "chunks.json") as f:
    chunks = json.load(f)


def search(question: str, top_k: int = 3) -> list[dict]:
    """Return the top_k chunks most similar to the question.

    Each result has "source" (short document name), "page", "score" (cosine
    similarity) and "text".
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q = np.array(response.embeddings[0].values)

    # Cosine similarity against every chunk in one vectorized operation.
    scores = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in best:
        c = chunks[i]
        results.append({
            "source": SOURCE_NAMES.get(c["source"], c["source"]),
            "page": c["page"],
            "score": round(float(scores[i]), 3),
            "text": c["text"],
        })
    return results
