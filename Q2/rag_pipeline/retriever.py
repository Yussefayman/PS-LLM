import numpy as np
from rag_pipeline.indexer import Index


def retrieve(query: str, index: Index, k: int = 3, metric: str = "cosine") -> list[dict]:
    if metric == "cosine":
        query_emb = index.embedder.encode(query, normalize_embeddings=True)
    elif metric == "dot":
        query_emb = index.embedder.encode(query, normalize_embeddings=False)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    scores      = index.embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]

    return [
        {
            "id":       index.corpus[i]["id"],
            "text":     index.corpus[i]["text"],
            "score":    round(float(scores[i]), 4),
            "metadata": index.corpus[i].get("metadata", {}),
        }
        for i in top_indices
    ]