"""Indexing subsystem public exports.

Fail-fast deterministic import policy (no guarded / optional imports):
All underlying modules are required at runtime. Missing dependencies should
surface immediately as ImportError to avoid silent feature degradation.

Exports:
    VectorStore: Primary vector storage / embedding interface
    DocumentationIndexer: Indexes built Sphinx documentation content
    CodebaseIndexer: Indexes Python source code for symbol / reference search
    ContextManager: High-level helper that aggregates context retrieval

Utility:
    get_availability_status(): Retained for existing callers but now always
        reports components as available (errors remain None). This preserves
        the previous interface while honoring the strict import contract.
"""

from .documentation_indexer import DocumentationIndexer  # type: ignore
from .codebase_indexer import CodebaseIndexer  # type: ignore
from .context_manager import ContextManager  # type: ignore
from .consumables_indexer import ConsumablesIndexer  # type: ignore
from .vector_store import VectorStore  # type: ignore

__all__ = [
    "VectorStore",
    "DocumentationIndexer",
    "CodebaseIndexer",
    "ContextManager",
    "get_availability_status",
    "ConsumablesIndexer",
]


def get_availability_status():
    """Return component availability status.

    Returns:
        dict: A mapping of component names to availability metadata. Under the
            fail-fast policy all components are required, so every entry has
            ``available=True`` and ``error=None``. This function remains for
            backward compatibility with earlier optional-import callers.
    """
    return {
        "vector_store": {"available": True, "error": None},
        "documentation_indexer": {"available": True, "error": None},
        "codebase_indexer": {"available": True, "error": None},
        "context_manager": {"available": True, "error": None},
    }
