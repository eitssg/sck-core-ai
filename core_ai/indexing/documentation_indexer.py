from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
import hashlib
import os
import re

from bs4 import BeautifulSoup
import tiktoken

import core_logging as log
from .vector_store import VectorStore


logger = log


class DocumentationIndexer:
    """
    Index and search built Sphinx documentation for LLM context.

    This class scans HTML files produced by sck-core-docs, extracts meaningful
    text, chunks it deterministically by token count, and persists embeddings via
    the provided VectorStore. It also provides full-text semantic search over the
    indexed content.

    Contract:
    - Input: build_directory path (root of docs build), VectorStore implementation
    - Output: documents added to collection_name with chunk metadata
    - Errors: Non-fatal; individual file failures are logged and skipped
    - Success criteria: All discoverable HTML pages (minus static/sources) produce >= 0 chunks

    Example:
        indexer = DocumentationIndexer(
            build_directory="../sck-core-docs/build",
            vector_store=vector_store
        )
        result = indexer.index_all_documentation()
        hits = indexer.search_documentation("how to configure buckets", n_results=5)
    """

    def __init__(
        self,
        build_directory: str | Path,
        vector_store: VectorStore,
        collection_name: str = "documentation",
        max_chunk_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
    ) -> None:
        """
        Initialize documentation indexer.

        Args:
            build_directory: Path to sck-core-docs/build directory.
            vector_store: Vector store abstraction for embeddings.
            collection_name: Target collection name in the store.
            max_chunk_tokens: Max tokens per chunk (tiktoken based).
            chunk_overlap_tokens: Overlap tokens between chunks.
        """
        self.build_directory = Path(build_directory)
        self.vector_store = vector_store
        self.collection_name = collection_name

        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = max(0, min(chunk_overlap_tokens, max_chunk_tokens // 2))

        # Canonical doc sections (subdirectories in build/)
        self.doc_sections: Dict[str, Dict[str, Any]] = {
            "technical_reference": {"title": "Technical Reference", "priority": 1},
            "developer_guide": {"title": "Developer Guide", "priority": 2},
            "user_guide": {"title": "User Guide", "priority": 3},
        }

        logger.info(f"Documentation indexer initialized for: {self.build_directory}")

    def clear_index(self) -> None:
        """Remove all indexed documentation from the vector store."""
        try:
            self.vector_store.clear_collection(self.collection_name)
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection {self.collection_name}: {e}")

    def index_all_documentation(self, clear_before: bool = True) -> Dict[str, int]:
        """
        Index all known documentation sections under build_directory.

        Args:
            clear_before: If True, clears existing collection before indexing.

        Returns:
            Mapping of section_name -> number of chunks indexed.
        """
        if not self.build_directory.exists():
            logger.error(f"Build directory not found: {self.build_directory}")
            return {}

        if clear_before:
            self.clear_index()

        results: Dict[str, int] = {}
        total = 0

        for section_name, meta in self.doc_sections.items():
            section_path = self.build_directory / section_name
            if not section_path.exists():
                logger.warning(f"Section directory not found: {section_path}")
                results[section_name] = 0
                continue

            count = self._index_section(section_name, section_path, meta)
            results[section_name] = count
            total += count
            logger.info(f"Indexed {count} chunks from section '{section_name}'")

        logger.info(f"Total documentation indexed: {total} chunks")
        return results

    def search_documentation(
        self,
        query: str,
        section: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across indexed documentation.

        Args:
            query: Search text.
            section: Optional section filter (e.g., 'developer_guide').
            n_results: Max results.

        Returns:
            VectorStore search results (implementation-defined dicts).
        """
        where = {"section": section} if section else None
        return self.vector_store.search_documents(
            collection_name=self.collection_name,
            query=query,
            n_results=n_results,
            where=where,
        )

    def get_documentation_stats(self) -> Dict[str, Any]:
        """
        Get high-level stats for the documentation collection.

        Returns:
            Stats dict (document_count, sections, etc.) where available.
        """
        try:
            stats = self.vector_store.get_collection_stats()
            doc_stats = stats.get(self.collection_name, {})
            return {
                "total_chunks": doc_stats.get("document_count", 0),
                "sections": list(self.doc_sections.keys()),
                "build_directory": str(self.build_directory),
                "max_chunk_tokens": self.max_chunk_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
            }
        except Exception as e:
            logger.error(f"Error getting documentation stats: {e}")
            return {"error": str(e)}

    # -------- Internal helpers --------

    def _index_section(self, section_name: str, section_path: Path, section_meta: Dict[str, Any]) -> int:
        indexed = 0

        for html_path in self._iter_html_files(section_path):
            try:
                page_rel = self._relative_url(html_path)
                page_title, text = self._extract(html_path)

                if not text:
                    logger.debug(f"No text from {page_rel}; skipping")
                    continue

                chunks = self._chunk_text(text)
                if not chunks:
                    continue

                # Prepare docs
                file_rel_path = str(html_path.relative_to(self.build_directory)).replace("\\", "/")
                file_mtime = os.path.getmtime(html_path)
                file_hash = self._sha1(text)[:12]

                docs = []
                metas = []
                ids = []

                for i, body in enumerate(chunks):
                    body_hash = self._sha1(body)[:10]
                    doc_id = f"{section_name}:{file_rel_path}#c{i}-{body_hash}"

                    meta = {
                        "source": "documentation",
                        "section": section_name,
                        "section_title": section_meta.get("title", section_name),
                        "section_priority": section_meta.get("priority", 99),
                        "page_title": page_title or "Unknown Title",
                        "page_url": page_rel,
                        "file_path": file_rel_path,
                        "file_mtime": file_mtime,
                        "file_hash": file_hash,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_type": "html",
                        "token_count": len(self.encoding.encode(body)),
                    }

                    ids.append(doc_id)
                    docs.append(body)
                    metas.append(meta)

                success = self.vector_store.add_documents(
                    collection_name=self.collection_name,
                    documents=docs,
                    metadatas=metas,
                    ids=ids,
                )
                if success:
                    indexed += len(docs)
                    logger.debug(f"Indexed {len(docs)} chunks from {page_rel}")
                else:
                    logger.error(f"VectorStore rejected docs from {page_rel}")

            except Exception as e:
                logger.error(f"Error processing {html_path}: {e}")

        return indexed

    def _iter_html_files(self, root: Path) -> Iterator[Path]:
        for p in root.rglob("*.html"):
            if p.name in {"genindex.html", "search.html", "404.html"}:
                continue
            if any(x in p.parts for x in ("_static", "_sources", "_templates")):
                continue
            yield p

    def _extract(self, html_path: Path) -> tuple[str, str]:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        title = self._extract_title(soup)
        text = self._extract_main_text(soup)
        return title, text

    def _extract_title(self, soup: BeautifulSoup) -> str:
        selectors = [
            "h1.sck-page-title",
            "h1",
            "title",
            ".document h1",
            ".body h1",
        ]
        for sel in selectors:
            node = soup.select_one(sel)
            if node:
                return node.get_text().strip()
        return "Unknown Title"

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        # Remove noise
        for sel in (
            "nav, header, footer, .navbar, .sidebar, .toctree-wrapper, "
            ".navigation, .breadcrumb, script, style, .highlight pre, code.literal"
        ).split(","):
            for node in soup.select(sel.strip()):
                node.decompose()

        main_candidates = [
            ".document .body",
            ".document",
            "main",
            ".content",
            "article",
            ".rst-content",
        ]
        main = None
        for sel in main_candidates:
            main = soup.select_one(sel)
            if main:
                break
        if not main:
            main = soup

        text = main.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _chunk_text(self, content: str) -> List[str]:
        if not content:
            return []

        ids = self.encoding.encode(content)
        if len(ids) <= self.max_chunk_tokens:
            return [content]

        chunks: List[str] = []
        start = 0
        while start < len(ids):
            end = min(start + self.max_chunk_tokens, len(ids))
            piece = self.encoding.decode(ids[start:end])

            # Prefer breaking at a sentence boundary for non-final chunks
            if end < len(ids):
                piece = self._break_at_sentence_boundary(piece)

            chunk = piece.strip()
            if chunk:
                chunks.append(chunk)

            # Move forward with overlap
            next_start = end - self.chunk_overlap_tokens
            if next_start <= start:  # safety
                next_start = end
            start = next_start

        return chunks

    def _break_at_sentence_boundary(self, text: str) -> str:
        if not text:
            return text
        pivot = int(len(text) * 0.8)
        tail = text[pivot:]
        endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        best = -1
        best_len = 0
        for e in endings:
            pos = tail.rfind(e)
            if pos > best:
                best = pos
                best_len = len(e.strip())
        if best > -1:
            return text[: pivot + best + best_len]
        return text

    def _relative_url(self, html_path: Path) -> str:
        try:
            rel = html_path.relative_to(self.build_directory)
            return str(rel).replace("\\", "/")
        except ValueError:
            return html_path.name

    @staticmethod
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
