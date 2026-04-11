import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL")


class Index:
    def __init__(self, corpus: list[dict], embeddings: np.ndarray, embedder: SentenceTransformer):
        self.corpus     = corpus
        self.embeddings = embeddings
        self.embedder   = embedder


def build_index(path: str ) -> Index:
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)

    embedder   = SentenceTransformer(MODEL_NAME)
    texts      = [c["text"] for c in corpus]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    return Index(corpus, embeddings, embedder)