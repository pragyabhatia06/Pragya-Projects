import os
import math
import hashlib


class EmbeddingService:
    """
    Local embedding model.
    No OpenAI key needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Avoid TensorFlow auto-paths when they are not needed for sentence embeddings.
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        self.dim = 384
        self.backend = "hash"
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
        except Exception:
            # Fallback keeps ingestion/search operational in constrained environments.
            self.model = None

    def _hash_embed(self, text: str) -> list:
        vec = [0.0] * self.dim
        tokens = (text or "").lower().split()

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = -1.0 if (digest[4] % 2) else 1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def embed_texts(self, texts: list) -> list:
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        return [self._hash_embed(text) for text in texts]

    def embed_query(self, query: str) -> list:
        if self.model is not None:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0].tolist()

        return self._hash_embed(query)