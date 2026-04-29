from typing import List, Dict

try:
    import chromadb
except ImportError:
    chromadb = None


class VectorStore:
    """
    ChromaDB vector store wrapper.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "llm_pdf_documents"
    ):
        if chromadb is None:
            raise ImportError(
                "Missing dependency 'chromadb'. Install it with: pip install chromadb"
            )

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """
        Store chunks with embeddings.
        """

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append({
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(ids)

    def search(self, query_embedding: List[float], top_k: int = 5) -> Dict:
        """
        Search similar chunks.
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results

    def count(self) -> int:
        return self.collection.count()

    def delete_collection_data(self):
        """
        Deletes all records by deleting and recreating collection.
        """

        try:
            self.client.delete_collection(name=self.collection.name)
        except Exception:
            # Keep behavior idempotent if collection was already removed.
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )