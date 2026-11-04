import json
import numpy as np
from pathlib import Path          
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# Friendly names for citations, so every agent sees the same short label
SOURCE_NAMES = {
    "data/Annual-Report-2025-eng.pdf": "Annual Report 2025",
    "data/audit-report-2025-eng-1777360685-.pdf": "Audited FS 2025",
}
# Load the index you built (fast, no API calls)
DATA = Path(__file__).parent.parent / "data"

matrix = np.load(DATA / "embeddings.npy")
with open(DATA / "chunks.json") as f:
    chunks = json.load(f)


def search(question: str, top_k: int = 3) -> list[dict]:
    # 1. TODO: embed the question as RETRIEVAL_QUERY, turn it into a numpy array q
    #    (same as lesson4a: client.models.embed_content(...), then .embeddings[0].values)
    q = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q = np.array(q.embeddings[0].values)     


    # 2. cosine similarity with ALL chunks at once (given, read the comment)
    #    matrix @ q            -> 1044 dot products
    #    np.linalg.norm(matrix, axis=1) -> length of each row (each chunk vector)
    scores = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))

    # 3. indexes of the top_k highest scores (given)
    #    argsort sorts low->high, [::-1] reverses it, [:top_k] takes the first few
    best = np.argsort(scores)[::-1][:top_k]

    # 4. TODO: build and return a list of dicts, one per index i in best:
    #    {"source": ..., "page": ..., "score": round(float(scores[i]), 3), "text": ...}
    #    (chunks[i] is the chunk dict for index i)
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


