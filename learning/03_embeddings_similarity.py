"""Lesson 3: embeddings and cosine similarity, the core of the RAG search.

Sentences that mean the same thing score high even with different wording, or
in another language (the Armenian sentence), while unrelated sentences score low.
"""

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

client = genai.Client()

question = "How much money did the bank earn?"

sentences = [
    "Net profit for the year was 42.5 billion AMD.",
    "Profit after tax increased by 12% compared to last year.",
    "The bank opened 5 new branches in Gyumri.",
    "The weather in Yerevan was sunny all week.",
    "Բանկի զուտ շահույթը կազմել է 42.5 միլիարդ դրամ։",  # Armenian: "The bank's net profit was 42.5 billion AMD"
]

# Queries and documents use different task types; the model optimizes each side for retrieval.
q = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[question],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
d = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
)

question_vec = np.array(q.embeddings[0].values)
sentence_vecs = [np.array(e.values) for e in d.embeddings]

print("Each embedding has", len(question_vec), "numbers")


def cosine_similarity(a, b):
    """Cosine of the angle between two vectors: 1 = same direction, 0 = unrelated."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


for text, vec in zip(sentences, sentence_vecs):
    score = cosine_similarity(question_vec, vec)
    print(round(score, 3), text)
