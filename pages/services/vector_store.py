import os
import shutil
from datetime import datetime
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

        self.persist_directory = persist_directory
        self.backend_mode = "persistent"
        self.init_error = None

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except Exception as exc:
            # Some managed runtimes fail to initialize persistent Chroma clients
            # due to tenant/rust binding initialization issues. Fall back safely.
            self.backend_mode = "ephemeral"
            self.init_error = str(exc)
            self.client = chromadb.EphemeralClient()

        self.collection_name = collection_name
        self.collection = self._create_collection_with_recovery()

    def _create_collection_with_recovery(self):
        try:
            return self.client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            error_text = str(exc)

            # Recover from persisted DB incompatibility/corruption (seen as KeyError: '_type').
            if self.backend_mode == "persistent" and "_type" in error_text:
                backup_path = f"{self.persist_directory}_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                try:
                    if os.path.exists(self.persist_directory):
                        shutil.move(self.persist_directory, backup_path)
                    os.makedirs(self.persist_directory, exist_ok=True)
                    self.client = chromadb.PersistentClient(path=self.persist_directory)
                    self.init_error = f"Recovered from corrupt/incompatible Chroma data. Backup: {backup_path}"
                    return self.client.get_or_create_collection(name=self.collection_name)
                except Exception as recovery_exc:
                    self.backend_mode = "ephemeral"
                    self.init_error = f"Persistent recovery failed ({recovery_exc}); using ephemeral backend."
                    self.client = chromadb.EphemeralClient()
                    return self.client.get_or_create_collection(name=self.collection_name)

            # For all other initialization failures, keep app usable via ephemeral backend.
            if self.backend_mode == "persistent":
                self.backend_mode = "ephemeral"
                self.init_error = f"Persistent backend failed ({error_text}); using ephemeral backend."
                self.client = chromadb.EphemeralClient()
                return self.client.get_or_create_collection(name=self.collection_name)

            raise

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