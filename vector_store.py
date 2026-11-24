"""
Vector Store Module
-------------------
Provides a clean, persistent vector database interface using ChromaDB.

Follows SOLID Principles:
- Single Responsibility: Manages only vector persistence, indexing, and similarity retrieval.
- Dependency Inversion: Exposes an abstract interface so other vector databases (e.g., Qdrant, FAISS) can be swapped in.
- Open/Closed: Extensible with custom distance metrics (cosine, l2, ip) and collections.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings
import numpy as np


# ============================================================================
# 1. Abstraction (Interface)
# ============================================================================

class BaseVectorStore(ABC):
    """Abstract interface defining required vector store operations."""

    @abstractmethod
    def add_document(
        self,
        document: str,
        embedding: Union[np.ndarray, List[float]],
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a single document with its embedding and metadata. If doc_id is None, generate a UUID."""
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        n_results: int = 3
    ) -> Dict[str, Any]:
        """Query the vector database using an embedding vector."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of documents in the collection."""
        pass


# ============================================================================
# 2. Concrete Persistent ChromaDB Implementation
# ============================================================================

class ChromaVectorStore(BaseVectorStore):
    """
    Persistent ChromaDB vector store implementation.
    Data is saved to disk and persists across application restarts.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "ceb_knowledge_base",
        distance_metric: str = "cosine"
    ) -> None:
        """
        Initialize the persistent ChromaDB client and collection.

        Args:
            persist_directory: Folder path on disk where data is saved.
            collection_name: Name of the collection/table.
            distance_metric: Distance function ('cosine', 'l2', or 'ip').
                             'cosine' is ideal for normalized embeddings like BGE-M3.
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name

        # Create persistent Chroma client (saves directly to disk)
        self._client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection with specified distance metric
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": distance_metric}
        )

    @property
    def collection(self):
        """Access underlying Chroma collection if needed."""
        return self._collection

    def add_document(
        self,
        document: str,
        embedding: Union[np.ndarray, List[float]],
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a single document, its vector embedding, and metadata into ChromaDB.
        If doc_id is omitted or None, a random UUIDv4 is automatically generated.

        Args:
            document: Raw text of the document.
            embedding: 1D vector (1024 floats from BGE-M3).
            doc_id: Optional unique string identifier. If None, auto-generates a UUID.
            metadata: Optional dictionary of key-value attributes (e.g. source, title).

        Returns:
            The document ID assigned (either provided or generated).
        """
        assigned_id = doc_id or str(uuid.uuid4())
        vector_list = self._normalize_vector(embedding)

        self._collection.upsert(
            ids=[assigned_id],
            documents=[document],
            embeddings=[vector_list],
            metadatas=[metadata or {}]
        )
        return assigned_id

    def clear(self) -> None:
        """Drop the entire collection and recreate an empty one to start completely fresh."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        doc_ids: List[str],
        documents: List[str],
        embeddings: Union[np.ndarray, List[List[float]]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Store multiple documents in a single batch.

        Args:
            doc_ids: List of unique string IDs.
            documents: List of text strings.
            embeddings: 2D array/list of vector embeddings.
            metadatas: Optional list of metadata dictionaries.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [self._normalize_vector(vec) for vec in embeddings]

        self._collection.upsert(
            ids=doc_ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas or [{} for _ in doc_ids]
        )

    def query(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Find the top most similar documents for a given query embedding.

        Args:
            query_embedding: Vector embedding of the user's question/query.
            n_results: Number of top similar results to return.

        Returns:
            Dictionary containing 'ids', 'documents', 'metadatas', and 'distances'.
        """
        vector_list = self._normalize_vector(query_embedding)

        return self._collection.query(
            query_embeddings=[vector_list],
            n_results=min(n_results, max(1, self.count()))
        )

    def count(self) -> int:
        """Return total document count stored in the collection."""
        return self._collection.count()

    @staticmethod
    def _normalize_vector(vector: Union[np.ndarray, List[float]]) -> List[float]:
        """Convert numpy arrays or nested shapes into a standard list of floats."""
        if isinstance(vector, np.ndarray):
            # Flatten in case shape is (1, 1024)
            return vector.flatten().tolist()
        return list(vector)
