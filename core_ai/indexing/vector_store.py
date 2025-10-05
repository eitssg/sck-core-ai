"""
Vector store management for document and code embeddings.

Handles ChromaDB collections, embedding generation, and semantic search
for documentation and codebase context retrieval.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, cast
import hashlib
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)

import core_logging as log

logger = log


class VectorStore:
    """
    Vector database for storing and searching document/code embeddings.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    Supports multiple collections for different content types.
    """

    def __init__(self, persist_directory: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            model_name: SentenceTransformer model name for embeddings
        """
        self.persist_directory = persist_directory or str(Path.cwd() / "data" / "vectordb")
        self.embedding_model_name = model_name

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Collections cache
        self.collections: Dict[str, Any] = {}

        # Default collection configs (lazy creation)
        self.collection_configs: Dict[str, Dict[str, Any]] = {
            "documentation": {
                "metadata": {"description": "Built documentation from Sphinx manuals"},
            },
            "codebase": {
                "metadata": {"description": "Source code from Python projects"},
            },
            "architecture": {
                "metadata": {"description": "Architecture and design documents"},
            },
            "consumables": {
                "metadata": {"description": "Specs, templates, and action consumables"},
            },
            "actions": {
                "metadata": {"description": "Discovered ActionResource entries"},
            },
        }

        logger.info(f"Vector store initialized (persist='{self.persist_directory}', model='{self.embedding_model_name}')")

    def ensure_collection(self, name: str):
        """Ensure a collection exists and is cached; create lazily if needed."""
        if name in self.collections:
            return self.collections[name]

        cfg = self.collection_configs.get(name, {"metadata": {"description": name}})
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=cast(Any, SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name)),
                metadata=cfg.get("metadata"),
            )
            self.collections[name] = collection
            logger.debug(f"Ensured collection: {name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to ensure collection {name}: {e}")
            raise

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
            ids: Optional list of document IDs (auto-generated if None)

        Returns:
            True if successful, False otherwise
        """
        # Ensure collection exists
        collection = self.ensure_collection(collection_name)

        try:
            # Sanitize metadatas for Chroma (no None / complex types)
            metadatas = [self._sanitize_metadata(m) for m in metadatas]

            # Generate IDs if not provided
            if ids is None:
                ids = [self._generate_document_id(doc, meta) for doc, meta in zip(documents, metadatas)]
            collection.add(documents=documents, metadatas=cast(Any, metadatas), ids=ids)

            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {e}")
            return False

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> bool:
        """Upsert documents into a collection, replacing on ID collision."""
        collection = self.ensure_collection(collection_name)
        try:
            # Sanitize metadatas for Chroma (no None / complex types)
            metadatas = [self._sanitize_metadata(m) for m in metadatas]
            # Prefer native upsert if available
            if hasattr(collection, "upsert"):
                collection.upsert(documents=documents, metadatas=cast(Any, metadatas), ids=ids)
            else:
                # Fallback: delete then add
                try:
                    collection.delete(ids=ids)
                except Exception:
                    pass
                collection.add(documents=documents, metadatas=cast(Any, metadatas), ids=ids)
            logger.info(f"Upserted {len(documents)} documents into {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert documents to {collection_name}: {e}")
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
            collection_name: Name of the collection to search
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of search results with documents, metadata, and distances
        """
        try:
            collection = self.ensure_collection(collection_name)
        except Exception:
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=cast(Any, ["documents", "metadatas", "distances"]),
            )

            # Format results
            formatted_results: List[Dict[str, Any]] = []
            docs_list = cast(Any, results.get("documents")) or []
            if not docs_list or not docs_list[0]:
                return []
            docs0 = docs_list[0]
            metas0 = results.get("metadatas") or [[]]
            metas0 = metas0[0] if metas0 and len(metas0) > 0 else [None] * len(docs0)
            dists0 = results.get("distances") or [[]]
            dists0 = dists0[0] if dists0 and len(dists0) > 0 else [None] * len(docs0)
            ids0 = results.get("ids") or [[]]
            ids0 = ids0[0] if ids0 and len(ids0) > 0 else [None] * len(docs0)

            for i in range(len(docs0)):
                formatted_results.append(
                    {
                        "document": docs0[i],
                        "metadata": metas0[i],
                        "distance": dists0[i],
                        "id": ids0[i],
                    }
                )

            logger.debug(f"Found {len(formatted_results)} results for query in {collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary mapping collection names to their statistics
        """
        stats = {}
        # Ensure default collections are present for reporting but don't force creation
        names = set(self.collections.keys()) | set(self.collection_configs.keys())
        for name in names:
            try:
                collection = self.ensure_collection(name)
            except Exception as e:
                logger.error(f"Failed to access collection {name}: {e}")
                stats[name] = {"error": str(e)}
                continue
            try:
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "metadata": collection.metadata,
                    "embedder": self.embedding_model_name,
                    "persist_directory": self.persist_directory,
                }
            except Exception as e:
                logger.error(f"Failed to get stats for {name}: {e}")
                stats[name] = {"error": str(e)}

        return stats

    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all documents from a collection.

        Args:
            collection_name: Name of the collection to clear

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete and recreate only the specified collection
            self.client.delete_collection(collection_name)
            # Remove cached instance if present
            if collection_name in self.collections:
                del self.collections[collection_name]
            # Recreate lazily
            self.ensure_collection(collection_name)
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            return False

    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """Delete specific documents by IDs from a collection."""
        try:
            collection = self.ensure_collection(collection_name)
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from {collection_name}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """List known collection names (configured or cached)."""
        names = set(self.collection_configs.keys()) | set(self.collections.keys())
        return sorted(names)

    def _generate_document_id(self, document: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a document based on content and metadata.

        Args:
            document: Document text
            metadata: Document metadata

        Returns:
            Unique document ID
        """
        # Create a hash based on content and key metadata
        content_hash = hashlib.md5(document.encode("utf-8", errors="ignore")).hexdigest()[:8]
        source = metadata.get("source", "unknown")
        file_path = metadata.get("file_path") or metadata.get("path") or "unknown"
        chunk_index = metadata.get("chunk_index", 0)

        # Compose deterministic, stable ID similar to DocumentationIndexer
        composed = f"{source}:{file_path}#c{chunk_index}-{content_hash}"
        return composed.replace("\\", "/").replace("/", "_")

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for ChromaDB: drop None values and coerce non-primitives to strings.

        Chroma expects metadata values to be str, int, float, or bool. This helper enforces
        that contract and avoids upsert/add failures when upstream indexers include None or
        complex types.
        """
        clean: Dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                try:
                    clean[k] = str(v)
                except Exception:
                    # As a last resort, skip unserializable values
                    continue
        return clean

    def get_similar_documents(self, collection_name: str, document_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a specific document.

        Args:
            collection_name: Name of the collection
            document_id: ID of the reference document
            n_results: Number of similar documents to return

        Returns:
            List of similar documents
        """
        try:
            collection = self.ensure_collection(collection_name)

            # Get the reference document
            ref_result = collection.get(ids=[document_id])
            if not ref_result or not ref_result.get("documents"):
                logger.error(f"Document {document_id} not found")
                return []

            # Use the reference document text as query
            ref_docs_list = cast(Any, ref_result.get("documents")) or []
            if not ref_docs_list:
                logger.error(f"Document {document_id} not found")
                return []
            ref_document = ref_docs_list[0]
            results = collection.query(
                query_texts=[ref_document],
                n_results=n_results + 1,  # +1 to exclude the reference document
                include=cast(Any, ["documents", "metadatas", "distances"]),
            )

            # Filter out the reference document and format results
            formatted_results: List[Dict[str, Any]] = []
            docs_list = cast(Any, results.get("documents")) or []
            if not docs_list or not docs_list[0]:
                return []
            docs0 = docs_list[0]
            metas0 = results.get("metadatas") or [[]]
            metas0 = metas0[0] if metas0 and len(metas0) > 0 else [None] * len(docs0)
            dists0 = results.get("distances") or [[]]
            dists0 = dists0[0] if dists0 and len(dists0) > 0 else [None] * len(docs0)
            ids0 = results.get("ids") or [[]]
            ids0 = ids0[0] if ids0 and len(ids0) > 0 else [None] * len(docs0)
            for i in range(len(docs0)):
                # Exclude the reference document by ID
                if ids0[i] == document_id:
                    continue
                formatted_results.append(
                    {
                        "document": docs0[i],
                        "metadata": metas0[i],
                        "distance": dists0[i],
                        "id": ids0[i],
                    }
                )
                if len(formatted_results) >= n_results:
                    break

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            return []
