import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load and preprocess dataset
df = pd.read_csv("data/dataset.csv", encoding="utf-8")
df = df.dropna(subset=["title", "desc", "genre", "rating", "pages", "totalratings"])

# Clean genre field
df["genre"] = df["genre"].apply(lambda g: ", ".join(set(g.split(","))))

# Create a combined field for embeddings
df["text"] = df["title"] + " - " + df["genre"] + " - " + df["desc"]

# Save the cleaned DataFrame
with open("model/book_data.pkl", "wb") as f:
    pickle.dump(df, f)

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# Save embeddings and FAISS index
np.save("model/embeddings.npy", embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
faiss.write_index(index, "model/book_index.faiss")

print(f"✅ Index built and saved. Total items: {len(df)}")
