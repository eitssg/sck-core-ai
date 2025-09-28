"""
Context manager for coordinating documentation and codebase search.

Provides unified interface for retrieving relevant context from both
documentation and codebase indexes for AI assistant queries.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

import core_logging as logger

from .vector_store import VectorStore
from .documentation_indexer import DocumentationIndexer
from .codebase_indexer import CodebaseIndexer


class ContextManager:
    """
    Unified context retrieval system for AI assistant queries.

    Coordinates searches across documentation and codebase indexes to provide
    comprehensive context for development assistance and code generation.
    """

    def __init__(
        self,
        build_directory: str,
        workspace_root: str,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize context manager.

        Args:
            build_directory: Path to sck-core-docs/build/ directory
            workspace_root: Root directory of the workspace
            vector_store: Optional existing vector store (creates new if None)
        """
        self.build_directory = Path(build_directory)
        self.workspace_root = Path(workspace_root)

        # Initialize or use provided vector store
        self.vector_store = vector_store or VectorStore()

        # Initialize indexers
        self.doc_indexer = DocumentationIndexer(
            build_directory=str(self.build_directory), vector_store=self.vector_store
        )

        self.code_indexer = CodebaseIndexer(
            workspace_root=str(self.workspace_root), vector_store=self.vector_store
        )

        # Context retrieval strategies
        self.retrieval_strategies = {
            "balanced": {"doc_weight": 0.5, "code_weight": 0.5},
            "documentation_focused": {"doc_weight": 0.8, "code_weight": 0.2},
            "code_focused": {"doc_weight": 0.2, "code_weight": 0.8},
            "documentation_only": {"doc_weight": 1.0, "code_weight": 0.0},
            "code_only": {"doc_weight": 0.0, "code_weight": 1.0},
        }

        logger.info("Context manager initialized")

    def initialize_indexes(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Initialize or rebuild all indexes.

        Args:
            force_rebuild: Whether to force rebuilding existing indexes

        Returns:
            Dictionary with initialization results
        """
        results = {"documentation": {}, "codebase": {}, "errors": []}

        try:
            # Check if indexes already exist and are populated
            stats = self.get_system_stats()
            needs_doc_index = (
                force_rebuild or stats["documentation"]["total_chunks"] == 0
            )
            needs_code_index = force_rebuild or stats["codebase"]["total_elements"] == 0

            # Index documentation if needed
            if needs_doc_index:
                logger.info("Indexing documentation...")
                doc_results = self.doc_indexer.index_all_documentation()
                results["documentation"] = doc_results
                logger.info(
                    f"Documentation indexing completed: {sum(doc_results.values())} chunks"
                )
            else:
                logger.info("Documentation index already exists, skipping")
                results["documentation"] = {
                    "status": "skipped",
                    "reason": "already_exists",
                }

            # Index codebase if needed
            if needs_code_index:
                logger.info("Indexing codebase...")
                code_results = self.code_indexer.index_all_projects()
                results["codebase"] = code_results
                logger.info(
                    f"Codebase indexing completed: {sum(code_results.values())} elements"
                )
            else:
                logger.info("Codebase index already exists, skipping")
                results["codebase"] = {"status": "skipped", "reason": "already_exists"}

        except Exception as e:
            error_msg = f"Error during index initialization: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

        return results

    def get_context_for_query(
        self,
        query: str,
        strategy: str = "balanced",
        max_results: int = 10,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Get relevant context for a query using specified retrieval strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy (balanced, documentation_focused, etc.)
            max_results: Maximum total results to return
            include_metadata: Whether to include detailed metadata

        Returns:
            Dictionary with context results from both sources
        """
        if strategy not in self.retrieval_strategies:
            logger.warning(f"Unknown strategy '{strategy}', using 'balanced'")
            strategy = "balanced"

        strategy_config = self.retrieval_strategies[strategy]

        # Calculate result allocation based on strategy
        doc_results_count = int(max_results * strategy_config["doc_weight"])
        code_results_count = int(max_results * strategy_config["code_weight"])

        # Ensure we get at least some results if weights are non-zero
        if strategy_config["doc_weight"] > 0 and doc_results_count == 0:
            doc_results_count = 1
        if strategy_config["code_weight"] > 0 and code_results_count == 0:
            code_results_count = 1

        context = {
            "query": query,
            "strategy": strategy,
            "documentation": [],
            "codebase": [],
            "summary": {"total_results": 0, "doc_results": 0, "code_results": 0},
        }

        try:
            # Search documentation if requested
            if doc_results_count > 0:
                doc_results = self.doc_indexer.search_documentation(
                    query=query, n_results=doc_results_count
                )
                context["documentation"] = self._format_results(
                    doc_results, include_metadata
                )
                context["summary"]["doc_results"] = len(doc_results)

            # Search codebase if requested
            if code_results_count > 0:
                code_results = self.code_indexer.search_codebase(
                    query=query, n_results=code_results_count
                )
                context["codebase"] = self._format_results(
                    code_results, include_metadata
                )
                context["summary"]["code_results"] = len(code_results)

            context["summary"]["total_results"] = (
                context["summary"]["doc_results"] + context["summary"]["code_results"]
            )

        except Exception as e:
            logger.error(f"Error retrieving context for query '{query}': {e}")
            context["error"] = str(e)

        return context

    def get_project_context(
        self, project_name: str, query: Optional[str] = None, max_results: int = 15
    ) -> Dict[str, Any]:
        """
        Get context specific to a project.

        Args:
            project_name: Name of the SCK project (e.g., sck-core-api)
            query: Optional search query within the project
            max_results: Maximum results to return

        Returns:
            Dictionary with project-specific context
        """
        context = {
            "project": project_name,
            "query": query,
            "documentation": [],
            "codebase": [],
        }

        try:
            # Search codebase for project-specific content
            code_results = self.code_indexer.search_codebase(
                query=query or f"project {project_name}",
                project=project_name,
                n_results=max_results,
            )
            context["codebase"] = self._format_results(
                code_results, include_metadata=True
            )

            # If there's a specific query, also search documentation
            if query:
                doc_results = self.doc_indexer.search_documentation(
                    query=f"{project_name} {query}", n_results=max_results // 2
                )
                context["documentation"] = self._format_results(
                    doc_results, include_metadata=True
                )

        except Exception as e:
            logger.error(f"Error retrieving project context for {project_name}: {e}")
            context["error"] = str(e)

        return context

    def get_architectural_context(
        self, query: str, max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Get architectural and design pattern context.

        Args:
            query: Architecture-related query
            max_results: Maximum results to return

        Returns:
            Dictionary with architectural context
        """
        # Architecture-focused search terms
        arch_terms = [
            "architecture",
            "design",
            "pattern",
            "framework",
            "structure",
            "lambda",
            "s3",
            "api",
            "authentication",
            "oauth",
            "envelope",
        ]

        # Enhance query with architectural terms
        enhanced_query = f"{query} " + " ".join(arch_terms[:3])

        return self.get_context_for_query(
            query=enhanced_query,
            strategy="documentation_focused",  # Architecture docs are primary
            max_results=max_results,
            include_metadata=True,
        )

    def search_by_element_type(
        self,
        element_type: str,
        query: Optional[str] = None,
        project: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for specific code element types.

        Args:
            element_type: Type of element (class, function, method, module)
            query: Optional search query
            project: Optional project filter
            max_results: Maximum results to return

        Returns:
            List of matching code elements
        """
        search_query = query or element_type

        try:
            results = self.code_indexer.search_codebase(
                query=search_query,
                project=project,
                element_type=element_type,
                n_results=max_results,
            )
            return self._format_results(results, include_metadata=True)

        except Exception as e:
            logger.error(f"Error searching for {element_type} elements: {e}")
            return []

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system-wide statistics
        """
        try:
            doc_stats = self.doc_indexer.get_documentation_stats()
            code_stats = self.code_indexer.get_codebase_stats()
            vector_stats = self.vector_store.get_collection_stats()

            return {
                "documentation": doc_stats,
                "codebase": code_stats,
                "vector_store": vector_stats,
                "build_directory": str(self.build_directory),
                "workspace_root": str(self.workspace_root),
                "available_strategies": list(self.retrieval_strategies.keys()),
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

    def rebuild_indexes(self) -> Dict[str, Any]:
        """
        Force rebuild of all indexes.

        Returns:
            Dictionary with rebuild results
        """
        logger.info("Starting full index rebuild...")
        return self.initialize_indexes(force_rebuild=True)

    def _format_results(
        self, results: List[Dict[str, Any]], include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Format search results for consistent output.

        Args:
            results: Raw search results
            include_metadata: Whether to include full metadata

        Returns:
            Formatted results list
        """
        formatted = []

        for result in results:
            formatted_result = {
                "content": result.get("document", ""),
                "relevance_score": 1.0
                - result.get("distance", 1.0),  # Convert distance to relevance
                "id": result.get("id", ""),
            }

            if include_metadata:
                metadata = result.get("metadata", {})
                formatted_result.update(
                    {
                        "source": metadata.get("source", "unknown"),
                        "type": metadata.get(
                            "element_type", metadata.get("section", "unknown")
                        ),
                        "location": metadata.get(
                            "file_path", metadata.get("page_url", "")
                        ),
                        "project": metadata.get("project", metadata.get("section", "")),
                        "line_number": metadata.get("line_number"),
                        "metadata": metadata,
                    }
                )

            formatted.append(formatted_result)

        return formatted
