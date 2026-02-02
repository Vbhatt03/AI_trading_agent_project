import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MemoryStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.memories = []

    def add(self, text):
        emb = self.embedder.encode([text])
        self.index.add(np.array(emb).astype("float32"))
        self.memories.append(text)

    def retrieve(self, query_text, k=2):
        emb = self.embedder.encode([query_text])
        D, I = self.index.search(np.array(emb).astype("float32"), k)
        # Only return valid indices (non-negative and within bounds)
        return [self.memories[i] for i in I[0] if 0 <= i < len(self.memories)]
memory = MemoryStore()
