"""Simple in-process vector store (semantic search) abstraction.

This module provides a lightweight alternative to heavier vector DBs (e.g.
Chroma, Qdrant) for local development and CI where installing compiled
dependencies can be painful (notably on Windows).

The original draft referenced a constant ``SKLEARN_AVAILABLE`` that was never
defined, causing a ``NameError`` whenever the store was instantiated. This
patch introduces an explicit import sentinel and surfaces a clear error with
actionable guidance when optional dependencies are missing.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

import core_logging as logger

# ---------------------------------------------------------------------------
# Optional dependency import guard
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly when dependencies installed
    import numpy as np  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # Broad except acceptable for optional dependency gate
    SKLEARN_AVAILABLE = False


class SimpleVectorStore:
    """
    Simple vector database using scikit-learn for similarity search.

    Uses sentence-transformers for embeddings and sklearn for cosine similarity.
    Provides basic functionality compatible with ChromaDB interface.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize simple vector store.

        Args:
            persist_directory: Directory to persist data (optional)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Vector search disabled: install dependencies with 'pip install scikit-learn sentence-transformers'."
            )

        self.persist_directory = persist_directory or str(
            Path.cwd() / "data" / "vectordb"
        )
        self.embedding_model_name = "all-MiniLM-L6-v2"  # Lightweight, fast model

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # In-memory storage for collections
        self.collections = {}

        # Ensure persist directory exists
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Load existing collections if available
        self._load_collections()

        logger.info(
            f"Simple vector store initialized with persist directory: {self.persist_directory}"
        )

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "SimpleCollection":
        """
        Get or create a collection.

        Args:
            name: Collection name
            metadata: Optional metadata

        Returns:
            Collection object
        """
        if name not in self.collections:
            self.collections[name] = SimpleCollection(
                name=name, embedding_model=self.embedding_model, metadata=metadata or {}
            )
            logger.debug(f"Created collection: {name}")

        return self.collections[name]

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary mapping collection names to their statistics
        """
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = {
                "document_count": collection.count(),
                "metadata": collection.metadata,
            }

        return stats

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Add documents to a collection.

        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [
                    self._generate_document_id(doc, meta)
                    for doc, meta in zip(documents, metadatas)
                ]

            collection.add(documents=documents, metadatas=metadatas, ids=ids)

            # Persist if directory is set
            if self.persist_directory:
                self._save_collection(collection_name)

            logger.info(
                f"Added {len(documents)} documents to collection {collection_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {e}")
            return False

    def search_documents(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents in a collection.

        Args:
            collection_name: Name of collection to search
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of search results with documents, metadata, and distances
        """
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []

        try:
            collection = self.collections[collection_name]
            return collection.query(query=query, n_results=n_results, where=where)

        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            return []

    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all documents from a collection.

        Args:
            collection_name: Name of collection to clear

        Returns:
            True if successful, False otherwise
        """
        try:
            if collection_name in self.collections:
                self.collections[collection_name].clear()

                # Remove persisted file
                if self.persist_directory:
                    collection_file = (
                        Path(self.persist_directory) / f"{collection_name}.pkl"
                    )
                    if collection_file.exists():
                        collection_file.unlink()

                logger.info(f"Cleared collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            return False

    def _generate_document_id(self, document: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for a document."""
        content_hash = hashlib.md5(document.encode()).hexdigest()[:8]
        source = metadata.get("source", "unknown")
        path = metadata.get("path", "unknown")

        return f"{source}_{path}_{content_hash}".replace("/", "_").replace("\\", "_")

    def _save_collection(self, collection_name: str):
        """Save collection to disk."""
        if not self.persist_directory:
            return

        try:
            collection = self.collections[collection_name]
            collection_file = Path(self.persist_directory) / f"{collection_name}.pkl"

            with open(collection_file, "wb") as f:
                pickle.dump(collection.to_dict(), f)

        except Exception as e:
            logger.error(f"Failed to save collection {collection_name}: {e}")

    def _load_collections(self):
        """Load collections from disk."""
        if not self.persist_directory or not Path(self.persist_directory).exists():
            return

        try:
            for collection_file in Path(self.persist_directory).glob("*.pkl"):
                collection_name = collection_file.stem

                with open(collection_file, "rb") as f:
                    collection_data = pickle.load(f)

                collection = SimpleCollection(
                    name=collection_name,
                    embedding_model=self.embedding_model,
                    metadata=collection_data.get("metadata", {}),
                )
                collection.from_dict(collection_data)
                self.collections[collection_name] = collection

                logger.debug(f"Loaded collection: {collection_name}")

        except Exception as e:
            logger.warning(f"Failed to load collections: {e}")


class SimpleCollection:
    """
    Simple collection implementation for document storage and search.
    """

    def __init__(
        self, name: str, embedding_model: SentenceTransformer, metadata: Dict[str, Any]
    ):
        """
        Initialize collection.

        Args:
            name: Collection name
            embedding_model: Sentence transformer model
            metadata: Collection metadata
        """
        self.name = name
        self.embedding_model = embedding_model
        self.metadata = metadata

        # Storage
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.embeddings = None

    def add(
        self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]
    ):
        """Add documents to collection."""
        # Generate embeddings
        new_embeddings = self.embedding_model.encode(documents)

        # Add to storage
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def query(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query collection for similar documents."""
        if len(self.documents) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Apply metadata filter if provided
        valid_indices = list(range(len(self.documents)))
        if where:
            valid_indices = []
            for i, metadata in enumerate(self.metadatas):
                if all(metadata.get(key) == value for key, value in where.items()):
                    valid_indices.append(i)

        if not valid_indices:
            return []

        # Get top results from valid indices
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, similarity in valid_similarities[:n_results]:
            results.append(
                {
                    "document": self.documents[i],
                    "metadata": self.metadatas[i],
                    "distance": 1.0 - similarity,  # Convert similarity to distance
                    "id": self.ids[i],
                }
            )

        return results

    def count(self) -> int:
        """Return number of documents in collection."""
        return len(self.documents)

    def clear(self):
        """Clear all documents from collection."""
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.embeddings = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary for persistence."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "ids": self.ids,
            "embeddings": (
                self.embeddings.tolist() if self.embeddings is not None else None
            ),
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load collection from dictionary."""
        self.documents = data.get("documents", [])
        self.metadatas = data.get("metadatas", [])
        self.ids = data.get("ids", [])

        embeddings_data = data.get("embeddings")
        if embeddings_data:
            self.embeddings = np.array(embeddings_data)
        else:
            self.embeddings = None
