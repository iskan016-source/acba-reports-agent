from pypdf import PdfReader

import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

files = ["data/Annual-Report-2025-eng.pdf", "data/audit-report-2025-eng-1777360685-.pdf"]

pages = []   # will hold one dict per page: {"source": ..., "page": ..., "text": ...}

for path in files:
    reader = PdfReader(path)
    # print(path, "has", len(reader.pages), "pages")

    # TODO 1: loop over reader.pages and for EACH page append a dict to `pages`:
    #   {"source": path, "page": <page number starting at 1>, "text": <page text>}
    #   - page text:  page.extract_text()
    #   - page number: use enumerate(reader.pages, start=1)
    #       -> gives (1, page1), (2, page2), ...  like zip with a counter
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages.append({"source":path, "page":number, "text": text})
  


# print("Total pages:", len(pages))

# TODO 2: find the pages that mention "Profit for the year"
#   loop over pages; if the phrase is in p["text"], print the source and page number

# for p in pages:
#     if "Profit for the year" in p["text"]:
#         print(p["source"], p["page"])

# print(pages[349]["text"])

def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    """Splits text into pieces of `size` characters; each piece repeats
    the last `overlap` characters of the previous one."""
    chunks = []
    # 1. start at position 0
    # 2. while the start position is before the end of the text:
    #      - take the slice text[start : start + size]  and append it
    #      - move start forward by (size - overlap)
    # 3. return the list
    start = 0
    while start<len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap


    return chunks




chunks = []   # one dict per chunk: {"source": ..., "page": ..., "text": ...}

for p in pages:
    # 1. some pages are images or blank, so extract_text() gives "" or only spaces.
    #    skip those. Hint: p["text"].strip() removes spaces; an empty string is "falsy"
    #    -> if not p["text"].strip():  continue
    # TODO A
    if not p["text"].strip():
        continue

    # 2. split this page's text into pieces with chunk_text
    # TODO B
    pieces = chunk_text(p["text"], 1500, 200)

    # 3. for EACH piece, append a dict with the page's source, page number, and the piece
    # TODO C  (a small loop inside this loop, same append pattern as before)
    for piece in pieces:

        chunks.append({"source": p["source"], "page": p["page"], "text": piece })


# print("Total chunks:", len(chunks))
# print(chunks[0])


load_dotenv()
client = genai.Client()

BATCH = 20
all_vectors = []

# Same idea as your chunk loop: jump forward 20 at a time
for i in range(0, len(chunks), BATCH):
    batch = chunks[i : i + BATCH]            # the next 20 chunk dicts
    texts = [c["text"] for c in batch]       # just their text (list comprehension!)

    # TODO 1: embed `texts` with gemini-embedding-001 as RETRIEVAL_DOCUMENT
    #         (you did this exact call in lesson4a; reuse it)
    result = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    # TODO 2: turn each embedding into a numpy array and add them ALL to all_vectors
    #   hint: all_vectors.extend( [ ... for e in result.embeddings ] )
    #   (.extend adds every item of a list; .append would add the list as ONE item)
    
    all_vectors.extend(np.array(e.values) for e in result.embeddings)


    print(f"embedded {i + len(batch)} / {len(chunks)}")

# Stack all vectors into one big table: rows = chunks, columns = 3072 numbers
matrix = np.array(all_vectors)
print("Matrix shape:", matrix.shape)

np.save("data/embeddings.npy", matrix)
with open("data/chunks.json", "w") as f:
    json.dump(chunks, f)
print("Saved!")