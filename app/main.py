# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import faiss
import pickle
import re
import random
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import gdown
import os

def download_models():
    model_files = {
        "book_data.pkl":     "18Kvy077vd4gm981cMqhW7jyvDdylSbhL",
        "embeddings.npy":    "1td9bNMTi0xGLq76x-2X6U7VKLglzI0dA",
        "book_index.faiss":  "1HRxjQf9VJ657HfX2GCD2v4zpOS2EpIjS",
    }

    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename}...")
            gdown.download(url, filename, quiet=False)
        else:
            print(f"{filename} already exists. Skipping download.")


download_models()
# ----- Load model and data -----
with open("book_data.pkl", "rb") as f:
    df = pickle.load(f)

embeddings = np.load("embeddings.npy")
index = faiss.read_index("book_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----- FastAPI app -----
app = FastAPI()


# Allow requests from frontend 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use frontend deployed URL
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
