from dotenv import load_dotenv
load_dotenv()

import numpy as np
from google import genai
from google.genai import types


client = genai.Client()

question = "How much money did the bank earn?"

sentences = [
    "Net profit for the year was 42.5 billion AMD.",
    "Profit after tax increased by 12% compared to last year.",
    "The bank opened 5 new branches in Gyumri.",
    "The weather in Yerevan was sunny all week.",
    "Բանկի զուտ շահույթը կազմել է 42.5 միլիարդ դրամ։",  # Armenian: "The bank's net profit was 42.5 billion AMD"
]

# embed the question as a QUERY
q = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[question],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)

# embed the sentences as DOCUMENTS
d = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
)
# 1. Get embeddings for the question + all sentences in ONE call

question_vec = np.array(q.embeddings[0].values)      # first one is the question
sentence_vecs = [np.array(e.values) for e in d.embeddings]     # the rest are the sentences

print("Each embedding has", len(question_vec), "numbers")

# TODO 1: write cosine_similarity(a, b) using the formula above
#   numpy helpers: np.dot(a, b)  and  np.linalg.norm(a)
def cosine_similarity(a, b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

# TODO 2: loop over sentences + sentence_vecs together (hint: zip)
#   and print the similarity score (rounded to 3 decimals) next to each sentence
for text, vec in zip(sentences, sentence_vecs):
    score = cosine_similarity(question_vec, vec)   # question vs THIS sentence
    print(round(score, 3), text)