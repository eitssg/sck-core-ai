"""
Vector store management for document and code embeddings.

Handles ChromaDB collections, embedding generation, and semantic search
for documentation and codebase context retrieval.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

import chromadb
from chromadb.config import Settings
import chromadb.utils
import chromadb.utils.embedding_functions
import chromadb.utils.embedding_functions.sentence_transformer_embedding_function
from sentence_transformers import SentenceTransformer

import core_logging as logger


class VectorStore:
    """
    Vector database for storing and searching document/code embeddings.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    Supports multiple collections for different content types.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """

        self.persist_directory = persist_directory or str(
            Path.cwd() / "data" / "vectordb"
        )
        self.embedding_model_name = "all-MiniLM-L6-v2"  # Lightweight, fast model

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Collections for different content types
        self.collections = {}
        self._initialize_collections()

        logger.info(
            f"Vector store initialized with persist directory: {self.persist_directory}"
        )

    def _initialize_collections(self):
        """Initialize ChromaDB collections for different content types."""
        collection_configs = {
            "documentation": {
                "metadata": {"description": "Built documentation from Sphinx manuals"},
                "embedding_function": chromadb.utils.embedding_functions.sentence_transformer_embedding_function.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                ),
            },
            "codebase": {
                "metadata": {"description": "Source code from Python projects"},
                "embedding_function": chromadb.utils.embedding_functions.sentence_transformer_embedding_function.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                ),
            },
            "architecture": {
                "metadata": {"description": "Architecture and design documents"},
                "embedding_function": chromadb.utils.embedding_functions.sentence_transformer_embedding_function.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                ),
            },
        }

        for name, config in collection_configs.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=config["embedding_function"],
                    metadata=config["metadata"],
                )
                self.collections[name] = collection
                logger.debug(f"Initialized collection: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize collection {name}: {e}")

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
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False

        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [
                    self._generate_document_id(doc, meta)
                    for doc, meta in zip(documents, metadatas)
                ]

            collection = self.collections[collection_name]
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

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
            collection_name: Name of the collection to search
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
            results = collection.query(
                query_texts=[query], n_results=n_results, where=where
            )

            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "id": results["ids"][0][i],
                    }
                )

            logger.debug(
                f"Found {len(formatted_results)} results for query in {collection_name}"
            )
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
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {"document_count": count, "metadata": collection.metadata}
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
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False

        try:
            # Delete and recreate collection to clear it
            self.client.delete_collection(collection_name)
            self._initialize_collections()
            logger.info(f"Cleared collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            return False

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
        content_hash = hashlib.md5(document.encode()).hexdigest()[:8]
        source = metadata.get("source", "unknown")
        path = metadata.get("path", "unknown")

        return f"{source}_{path}_{content_hash}".replace("/", "_").replace("\\", "_")

    def get_similar_documents(
        self, collection_name: str, document_id: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a specific document.

        Args:
            collection_name: Name of the collection
            document_id: ID of the reference document
            n_results: Number of similar documents to return

        Returns:
            List of similar documents
        """
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []

        try:
            collection = self.collections[collection_name]

            # Get the reference document
            ref_result = collection.get(ids=[document_id])
            if not ref_result["documents"]:
                logger.error(f"Document {document_id} not found")
                return []

            # Use the reference document text as query
            ref_document = ref_result["documents"][0]
            results = collection.query(
                query_texts=[ref_document],
                n_results=n_results + 1,  # +1 to exclude the reference document
            )

            # Filter out the reference document and format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                if results["ids"][0][i] != document_id:  # Exclude reference document
                    formatted_results.append(
                        {
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "id": results["ids"][0][i],
                        }
                    )

                    if len(formatted_results) >= n_results:
                        break

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            return []
