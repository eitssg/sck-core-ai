"""
Documentation indexer for extracting and processing built Sphinx documentation.

Processes HTML files from sck-core-docs/build/ directory to create searchable
embeddings for technical reference, user guide, and developer guide content.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

import tiktoken

import core_logging as logger

from .vector_store import VectorStore


class DocumentationIndexer:
    """
    Indexes built Sphinx documentation for semantic search.

    Processes HTML files from the documentation build directory,
    extracts meaningful content, and creates searchable embeddings.
    """

    def __init__(self, build_directory: str, vector_store: VectorStore):
        """
        Initialize documentation indexer.

        Args:
            build_directory: Path to sck-core-docs/build/ directory
            vector_store: Vector store for embeddings
        """

        self.build_directory = Path(build_directory)
        self.vector_store = vector_store

        # Token counting for content chunking
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        self.max_chunk_tokens = 512  # Reasonable chunk size for embeddings
        self.chunk_overlap_tokens = 50  # Overlap between chunks

        # Documentation structure mapping
        self.doc_sections = {
            "technical_reference": {
                "title": "Technical Reference",
                "description": "API documentation and module references",
                "priority": 1,
            },
            "developer_guide": {
                "title": "Developer Guide",
                "description": "Contributing and development guidelines",
                "priority": 2,
            },
            "user_guide": {
                "title": "User Guide",
                "description": "End-user documentation and tutorials",
                "priority": 3,
            },
        }

        logger.info(f"Documentation indexer initialized for: {self.build_directory}")

    def index_all_documentation(self) -> Dict[str, int]:
        """
        Index all documentation sections.

        Returns:
            Dictionary mapping section names to document counts
        """
        if not self.build_directory.exists():
            logger.error(f"Build directory not found: {self.build_directory}")
            return {}

        results = {}
        total_indexed = 0

        # Clear existing documentation collection
        self.vector_store.clear_collection("documentation")

        for section_name, section_info in self.doc_sections.items():
            section_path = self.build_directory / section_name

            if not section_path.exists():
                logger.warning(f"Section directory not found: {section_path}")
                results[section_name] = 0
                continue

            indexed_count = self._index_section(
                section_name, section_path, section_info
            )
            results[section_name] = indexed_count
            total_indexed += indexed_count

            logger.info(f"Indexed {indexed_count} documents from {section_name}")

        logger.info(f"Total documentation indexed: {total_indexed} chunks")
        return results

    def _index_section(
        self, section_name: str, section_path: Path, section_info: Dict[str, Any]
    ) -> int:
        """
        Index a specific documentation section.

        Args:
            section_name: Name of the documentation section
            section_path: Path to the section directory
            section_info: Section metadata

        Returns:
            Number of document chunks indexed
        """
        indexed_count = 0

        for html_file in self._find_html_files(section_path):
            try:
                chunks = self._process_html_file(html_file, section_name, section_info)

                if chunks:
                    # Prepare data for vector store
                    documents = [chunk["content"] for chunk in chunks]
                    metadatas = [chunk["metadata"] for chunk in chunks]
                    ids = [chunk["id"] for chunk in chunks]

                    # Add to vector store
                    success = self.vector_store.add_documents(
                        collection_name="documentation",
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                    )

                    if success:
                        indexed_count += len(chunks)
                        logger.debug(
                            f"Indexed {len(chunks)} chunks from {html_file.name}"
                        )
                    else:
                        logger.error(f"Failed to index chunks from {html_file.name}")

            except Exception as e:
                logger.error(f"Error processing {html_file}: {e}")
                continue

        return indexed_count

    def _find_html_files(self, directory: Path) -> Iterator[Path]:
        """
        Find all HTML files in a directory recursively.

        Args:
            directory: Directory to search

        Yields:
            Path objects for HTML files
        """
        for html_file in directory.rglob("*.html"):
            # Skip special files
            if html_file.name in ["genindex.html", "search.html", "404.html"]:
                continue

            # Skip directories that typically contain non-content files
            if any(
                skip in html_file.parts
                for skip in ["_static", "_sources", "_templates"]
            ):
                continue

            yield html_file

    def _process_html_file(
        self, html_file: Path, section_name: str, section_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process an HTML file and extract content chunks.

        Args:
            html_file: Path to HTML file
            section_name: Name of the documentation section
            section_info: Section metadata

        Returns:
            List of content chunks with metadata
        """
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract page metadata
            page_title = self._extract_page_title(soup)
            page_url = self._get_relative_url(html_file)

            # Extract main content
            content_text = self._extract_content_text(soup)

            if not content_text.strip():
                logger.debug(f"No content extracted from {html_file.name}")
                return []

            # Split content into chunks
            chunks = self._split_content_into_chunks(content_text)

            # Create chunk metadata
            chunk_list = []
            for i, chunk_content in enumerate(chunks):
                chunk_id = f"{section_name}_{html_file.stem}_{i}"

                metadata = {
                    "source": "documentation",
                    "section": section_name,
                    "section_title": section_info["title"],
                    "section_priority": section_info["priority"],
                    "page_title": page_title,
                    "page_url": page_url,
                    "file_path": str(html_file.relative_to(self.build_directory)),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "html",
                    "token_count": len(self.encoding.encode(chunk_content)),
                }

                chunk_list.append(
                    {"id": chunk_id, "content": chunk_content, "metadata": metadata}
                )

            return chunk_list

        except Exception as e:
            logger.error(f"Error processing HTML file {html_file}: {e}")
            return []

    def _extract_page_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try different title sources in order of preference
        title_selectors = [
            "h1.sck-page-title",
            "h1",
            "title",
            ".document h1",
            ".body h1",
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()

        return "Unknown Title"

    def _extract_content_text(self, soup: BeautifulSoup) -> str:
        """
        Extract main content text from HTML, excluding navigation and sidebars.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Cleaned content text
        """
        # Remove unwanted elements
        unwanted_selectors = [
            "nav",
            "header",
            "footer",
            ".navbar",
            ".sidebar",
            ".toctree-wrapper",
            ".navigation",
            ".breadcrumb",
            "script",
            "style",
            ".highlight pre",
            "code.literal",
        ]

        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Try to find main content area
        main_content_selectors = [
            ".document .body",
            ".document",
            "main",
            ".content",
            "article",
            ".rst-content",
        ]

        content_element = None
        for selector in main_content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break

        if not content_element:
            content_element = soup

        # Extract text with some formatting preservation
        text = content_element.get_text(separator=" ", strip=True)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def _get_relative_url(self, html_file: Path) -> str:
        """
        Get relative URL for the HTML file within the documentation.

        Args:
            html_file: Path to HTML file

        Returns:
            Relative URL string
        """
        try:
            relative_path = html_file.relative_to(self.build_directory)
            return str(relative_path).replace("\\", "/")
        except ValueError:
            return str(html_file.name)

    def _split_content_into_chunks(self, content: str) -> List[str]:
        """
        Split content into token-sized chunks with overlap.

        Args:
            content: Text content to split

        Returns:
            List of content chunks
        """
        if not content.strip():
            return []

        # Encode the full content
        tokens = self.encoding.encode(content)

        if len(tokens) <= self.max_chunk_tokens:
            return [content]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Define chunk boundaries
            end_idx = min(start_idx + self.max_chunk_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode chunk back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Try to break at sentence boundaries if possible
            if end_idx < len(tokens):  # Not the last chunk
                chunk_text = self._break_at_sentence_boundary(chunk_text)

            chunks.append(chunk_text.strip())

            # Move start position with overlap
            start_idx = end_idx - self.chunk_overlap_tokens

            # Prevent infinite loop
            if start_idx >= end_idx:
                break

        return [chunk for chunk in chunks if chunk.strip()]

    def _break_at_sentence_boundary(self, text: str) -> str:
        """
        Break text at the last sentence boundary to avoid cutting mid-sentence.

        Args:
            text: Text to break

        Returns:
            Text broken at sentence boundary
        """
        # Look for sentence endings in the last 20% of the text
        break_point = int(len(text) * 0.8)
        search_text = text[break_point:]

        # Find sentence endings
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]

        best_break = -1
        for ending in sentence_endings:
            pos = search_text.rfind(ending)
            if pos > best_break:
                best_break = pos

        if best_break > -1:
            # Break after the sentence ending
            actual_break = break_point + best_break + len(ending.rstrip())
            return text[:actual_break].strip()

        return text

    def search_documentation(
        self, query: str, section: Optional[str] = None, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search documentation content.

        Args:
            query: Search query
            section: Optional section filter
            n_results: Number of results to return

        Returns:
            List of search results
        """
        where_filter = None
        if section:
            where_filter = {"section": section}

        return self.vector_store.search_documents(
            collection_name="documentation",
            query=query,
            n_results=n_results,
            where=where_filter,
        )

    def get_documentation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed documentation.

        Returns:
            Dictionary with indexing statistics
        """
        try:
            collection_stats = self.vector_store.get_collection_stats()
            doc_stats = collection_stats.get("documentation", {})

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
