# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import faiss
import random
import os
import gdown
from sentence_transformers import SentenceTransformer

# ----- Download embeddings and index if not present -----
def download_files():
    model_files = {
        "embeddings.npy": "1C66l8N1vujIAUGyvYmVybNIYIgjn5lbe",
        "book_index.faiss": "1-7yNnm315Va9QpI0rCAjBWHHVIKvXjpg",
    }

    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename}...")
            gdown.download(url, filename, quiet=False)
        else:
            print(f"{filename} already exists. Skipping.")

download_files()

# ----- Load metadata from CSV -----
df = pd.read_csv("data/filtered_books.csv")

# ----- Load embeddings and FAISS index -----
embeddings = np.load("embeddings.npy")
index = faiss.read_index("book_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----- FastAPI app -----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Request model -----
class QueryRequest(BaseModel):
    query: str

# ----- Helper functions -----
def extract_length_preferences(text):
    excluded = set()
    text = text.lower()
    if "not long" in text or "short" in text:
        excluded.add("long")
    if "not short" in text:
        excluded.add("short")
    return excluded

def is_length_match(pages, excluded_terms):
    if "short" in excluded_terms and pages < 150:
        return False
    if "long" in excluded_terms and pages > 300:
        return False
    return True

# ----- Main recommend endpoint -----
@app.post("/recommend")
def recommend_books(request: QueryRequest):
    query = request.query
    excluded = extract_length_preferences(query)
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=50)

    matching_books = []
    for idx in I[0]:
        book = df.iloc[idx]
        pages = int(book["pages"]) if str(book["pages"]).isdigit() else 0

        if not is_length_match(pages, excluded):
            continue

        matching_books.append({
            "title": book["title"],
            "author": book["author"],
            "genre": book["genre"],
            "pages": pages,
            "index": int(idx),
            "img": book.get("img", "")
        })

    if matching_books:
        recommendations = random.sample(matching_books, min(7, len(matching_books)))
        return {"books": recommendations}
    else:
        return {"books": [], "message": "No matching books found"}
