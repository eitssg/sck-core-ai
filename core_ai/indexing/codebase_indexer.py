"""
Codebase indexer for analyzing and indexing Python source code.

Processes Python files across all sck-core-* projects to create searchable
embeddings for functions, classes, modules, and architectural patterns.
"""

from typing import Dict, List, Optional, Any, Iterator, Union
import os

import ast
from pathlib import Path
from .vector_store import VectorStore

import tiktoken

import core_logging as logger


class CodebaseIndexer:
    """
    Indexes Python source code for semantic search and architectural analysis.

    Processes Python files to extract functions, classes, docstrings, and
    architectural patterns for context-aware development assistance.
    """

    def __init__(self, workspace_root: str, vector_store: VectorStore):
        """
        Initialize codebase indexer.

        Args:
            workspace_root: Root directory of the workspace (simple-cloud-kit)
            vector_store: Vector store for embeddings
        """
        self.workspace_root = Path(workspace_root)
        self.vector_store = vector_store

        # Token counting for content chunking
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_chunk_tokens = 1024  # Larger chunks for code context
        self.chunk_overlap_tokens = 100

        # SCK project patterns
        self.sck_projects = [
            "sck-core-framework",
            "sck-core-db",
            "sck-core-execute",
            "sck-core-report",
            "sck-core-runner",
            "sck-core-deployspec",
            "sck-core-component",
            "sck-core-invoker",
            "sck-core-organization",
            "sck-core-api",
            "sck-core-codecommit",
            "sck-core-cli",
            "sck-core-ai",
        ]

        # File patterns to include/exclude
        self.include_patterns = {
            "*.py",  # Python source files
            "*.pyi",  # Python stub files
        }

        self.exclude_patterns = {
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".pytest_cache",
            "htmlcov",
            "build",
            "dist",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".tox",
            "*.egg-info",
        }

        # Code element types and their priorities
        self.element_priorities = {
            "module": 1,  # Module-level documentation
            "class": 2,  # Class definitions
            "function": 3,  # Function definitions
            "method": 4,  # Class methods
            "constant": 5,  # Module constants
            "import": 6,  # Import statements
            "comment": 7,  # Inline comments
        }

        logger.info(f"Codebase indexer initialized for workspace: {self.workspace_root}")

    def index_all_projects(self) -> Dict[str, int]:
        """
        Index all SCK projects in the workspace.

        Returns:
            Dictionary mapping project names to indexed element counts
        """
        results = {}
        total_indexed = 0

        # Clear existing codebase collection
        self.vector_store.clear_collection("codebase")

        for project_name in self.sck_projects:
            project_path = self.workspace_root / project_name

            if not project_path.exists():
                logger.warning(f"Project directory not found: {project_path}")
                results[project_name] = 0
                continue

            indexed_count = self._index_project(project_name, project_path)
            results[project_name] = indexed_count
            total_indexed += indexed_count

            logger.info(f"Indexed {indexed_count} code elements from {project_name}")

        logger.info(f"Total codebase indexed: {total_indexed} elements")
        return results

    def _index_project(self, project_name: str, project_path: Path) -> int:
        """
        Index a specific SCK project.

        Args:
            project_name: Name of the project (e.g., sck-core-api)
            project_path: Path to the project directory

        Returns:
            Number of code elements indexed
        """
        indexed_count = 0

        # Find Python source directories within the project
        source_dirs = self._find_source_directories(project_path)

        # 1) Index documentation (README + docs/**/*.{md,rst})
        try:
            indexed_count += self._index_project_docs(project_name, project_path)
        except Exception as e:
            logger.error(f"Error indexing docs for {project_name}: {e}")

        for source_dir in source_dirs:
            for python_file in self._find_python_files(source_dir):
                try:
                    elements = self._process_python_file(python_file, project_name)

                    if elements:
                        # Prepare data for vector store
                        documents = [elem["content"] for elem in elements]
                        metadatas = [elem["metadata"] for elem in elements]
                        ids = [elem["id"] for elem in elements]

                        # Upsert into vector store (idempotent re-index)
                        success = self.vector_store.upsert_documents(
                            collection_name="codebase",
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids,
                        )

                        if success:
                            indexed_count += len(elements)
                            logger.debug(f"Indexed {len(elements)} elements from {python_file.name}")
                        else:
                            logger.error(f"Failed to index elements from {python_file.name}")

                except Exception as e:
                    logger.error(f"Error processing {python_file}: {e}")
                    continue

        return indexed_count

    # ---------------- Documentation indexing ----------------
    def _index_project_docs(self, project_name: str, project_path: Path) -> int:
        indexed = 0
        for doc_path in self._find_doc_files(project_path):
            try:
                rel_path = self._safe_relative_path(doc_path)
                text = doc_path.read_text(encoding="utf-8", errors="ignore")
                chunks = self._chunk_text(text, self.max_chunk_tokens, self.chunk_overlap_tokens)

                is_readme = doc_path.name.lower().startswith("readme.")
                element_type = "readme" if is_readme else "doc"
                fmt = (
                    "md"
                    if doc_path.suffix.lower() == ".md"
                    else ("rst" if doc_path.suffix.lower() == ".rst" else doc_path.suffix.lower().lstrip("."))
                )
                title = self._extract_doc_title(text, fmt)

                documents = chunks
                metadatas = []
                ids = []
                for i, chunk in enumerate(chunks):
                    rel_posix = rel_path.as_posix() if isinstance(rel_path, Path) else str(rel_path).replace(os.sep, "/")
                    meta = {
                        "source": "codebase",
                        "project": project_name,
                        "file_path": str(rel_posix),
                        "element_type": element_type,
                        "title": title,
                        "format": fmt,
                        "chunk_index": i,
                        "token_count": len(self.encoding.encode(chunk)),
                    }
                    metadatas.append(meta)
                    ids.append(f"{project_name}__doc__{rel_posix}__chunk-{i}")

                if documents:
                    success = self.vector_store.upsert_documents(
                        collection_name="codebase",
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    if success:
                        indexed += len(documents)
                    else:
                        logger.error(f"Failed to index docs from {doc_path.name}")
            except Exception as e:
                logger.error(f"Error processing doc file {doc_path}: {e}")
        return indexed

    def _find_doc_files(self, project_path: Path) -> Iterator[Path]:
        names = {"README.md", "README.rst"}
        # Root-level README
        for n in names:
            p = project_path / n
            if p.exists() and p.is_file():
                yield p
        # Try package-dir README (e.g., sck-core-api/core_api/README.md)
        package_dir = project_path / project_path.name.replace("-", "_")
        for n in names:
            p = package_dir / n
            if p.exists() and p.is_file():
                yield p
        # docs/**/*.md|rst
        docs_dir = project_path / "docs"
        if docs_dir.exists() and docs_dir.is_dir():
            for ext in ("*.md", "*.rst"):
                for p in docs_dir.rglob(ext):
                    if self._is_excluded_path(p):
                        continue
                    if p.is_file():
                        yield p

    def _chunk_text(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        try:
            toks = self.encoding.encode(text)
        except Exception:
            # Fallback: no chunking if encoding fails
            return [text]
        if len(toks) <= max_tokens:
            return [text]
        chunks: List[str] = []
        step = max(1, max_tokens - max(0, overlap_tokens))
        for i in range(0, len(toks), step):
            chunk_tokens = toks[i : min(i + max_tokens, len(toks))]
            chunks.append(self.encoding.decode(chunk_tokens))
            if i + max_tokens >= len(toks):
                break
        return chunks

    def _extract_doc_title(self, text: str, fmt: str) -> Optional[str]:
        # Simple heuristic: first Markdown or RST heading
        if fmt == "md":
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("# ") or s.startswith("## ") or s.startswith("### "):
                    return s.lstrip("# ").strip()
                # Alternative underline style handled below
        elif fmt == "rst":
            lines = text.splitlines()
            for i in range(len(lines) - 1):
                title = lines[i].strip()
                underline = lines[i + 1].strip()
                if title and set(underline) <= set("=-~^`:") and len(underline) >= len(title):
                    return title
        # Fallback: first non-empty line
        for line in text.splitlines():
            if line.strip():
                return line.strip()[:120]
        return None

    def _is_excluded_path(self, path: Path) -> bool:
        s = str(path)
        return any(pat in s for pat in self.exclude_patterns)

    def _safe_relative_path(self, path: Path) -> Path:
        try:
            return path.relative_to(self.workspace_root)
        except ValueError:
            return path

    def _find_source_directories(self, project_path: Path) -> List[Path]:
        """
        Find Python source directories within a project.

        Args:
            project_path: Path to the project

        Returns:
            List of source directory paths
        """
        source_dirs = []

        # Look for common Python source directory patterns
        potential_dirs = [
            project_path,  # Root level
            project_path / "src",  # src/ layout
            project_path / project_path.name.replace("-", "_"),  # core_api/ layout
        ]

        # Add any directory that contains Python files and looks like source
        for item in project_path.iterdir():
            if item.is_dir() and not any(pattern in item.name for pattern in self.exclude_patterns):
                # Check if it contains Python files
                if any(item.rglob("*.py")):
                    potential_dirs.append(item)

        # Filter to existing directories with Python files
        for dir_path in potential_dirs:
            if dir_path.exists() and any(dir_path.rglob("*.py")):
                source_dirs.append(dir_path)

        return list(set(source_dirs))  # Remove duplicates

    def _find_python_files(self, directory: Path) -> Iterator[Path]:
        """
        Find Python files in a directory recursively.

        Args:
            directory: Directory to search

        Yields:
            Path objects for Python files
        """
        for py_file in directory.rglob("*.py"):
            # Skip excluded patterns
            if any(pattern in str(py_file) for pattern in self.exclude_patterns):
                continue

            # Skip empty files
            try:
                if py_file.stat().st_size == 0:
                    continue
            except OSError:
                continue

            yield py_file

    def _process_python_file(self, python_file: Path, project_name: str) -> List[Dict[str, Any]]:
        """
        Process a Python file and extract code elements.

        Args:
            python_file: Path to Python file
            project_name: Name of the project

        Returns:
            List of code elements with metadata
        """
        try:
            with open(python_file, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Parse the AST
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                logger.warning(f"Syntax error in {python_file}: {e}")
                return []

            # Extract elements from AST
            elements = []

            # Module-level docstring and metadata
            module_element = self._extract_module_info(python_file, project_name, tree, source_code)
            if module_element:
                elements.append(module_element)

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_element = self._extract_class_info(python_file, project_name, node, source_code)
                    if class_element:
                        elements.append(class_element)

                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    function_element = self._extract_function_info(python_file, project_name, node, source_code)
                    if function_element:
                        elements.append(function_element)

            return elements

        except Exception as e:
            logger.error(f"Error processing Python file {python_file}: {e}")
            return []

    def _extract_module_info(
        self, python_file: Path, project_name: str, tree: ast.Module, source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract module-level information.

        Args:
            python_file: Path to Python file
            project_name: Project name
            tree: AST tree
            source_code: Source code string

        Returns:
            Module element dictionary or None
        """
        # Get module docstring
        docstring = ast.get_docstring(tree)

        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        # Get relative path
        try:
            relative_path = python_file.relative_to(self.workspace_root)
        except ValueError:
            relative_path = python_file

        # Create content combining docstring and key information
        content_parts = []

        if docstring:
            content_parts.append(f"Module: {python_file.name}")
            content_parts.append(f"Documentation: {docstring}")

        if imports:
            content_parts.append(f"Key imports: {', '.join(sorted(set(imports))[:10])}")  # Top 10 imports

        # Add some context about the file structure
        lines = source_code.split("\n")
        if len(lines) > 10:
            # Add first few lines as context (usually contains important imports/constants)
            context_lines = [line.strip() for line in lines[:10] if line.strip() and not line.strip().startswith("#")]
            if context_lines:
                content_parts.append(f"File structure: {' | '.join(context_lines[:5])}")

        if not content_parts:
            return None

        content = "\n\n".join(content_parts)

        rel_posix = relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path).replace(os.sep, "/")
        element_id = f"{project_name}__module__{rel_posix}"

        metadata = {
            "source": "codebase",
            "project": project_name,
            "file_path": str(rel_posix),
            "element_type": "module",
            "element_name": python_file.stem,
            "priority": self.element_priorities["module"],
            "line_number": 1,
            "has_docstring": docstring is not None,
            "import_count": len(imports),
            "token_count": len(self.encoding.encode(content)),
        }

        return {"id": element_id, "content": content, "metadata": metadata}

    def _extract_class_info(
        self, python_file: Path, project_name: str, node: ast.ClassDef, source_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract class information.

        Args:
            python_file: Path to Python file
            project_name: Project name
            node: AST class node
            source_code: Source code string

        Returns:
            Class element dictionary or None
        """
        # Get class docstring
        docstring = ast.get_docstring(node)

        # Get method names
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.attr}")

        # Get relative path
        try:
            relative_path = python_file.relative_to(self.workspace_root)
        except ValueError:
            relative_path = python_file

        # Create content
        content_parts = []

        content_parts.append(f"Class: {node.name}")

        if bases:
            content_parts.append(f"Inherits from: {', '.join(bases)}")

        if docstring:
            content_parts.append(f"Documentation: {docstring}")

        if methods:
            content_parts.append(f"Methods: {', '.join(methods)}")

        # Add some source code context
        lines = source_code.split("\n")
        if hasattr(node, "lineno") and node.lineno <= len(lines):
            # Get a few lines of the class definition
            start_line = max(0, node.lineno - 1)
            context_lines = []
            for i in range(start_line, min(len(lines), start_line + 5)):
                line = lines[i].strip()
                if line and not line.startswith("#"):
                    context_lines.append(line)

            if context_lines:
                content_parts.append(f"Definition: {' | '.join(context_lines)}")

        content = "\n\n".join(content_parts)

        rel_posix = relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path).replace(os.sep, "/")
        lineno = getattr(node, "lineno", 0)
        element_id = f"{project_name}__class__{rel_posix}__{node.name}@{lineno}"

        metadata = {
            "source": "codebase",
            "project": project_name,
            "file_path": str(rel_posix),
            "element_type": "class",
            "element_name": node.name,
            "priority": self.element_priorities["class"],
            "line_number": getattr(node, "lineno", 0),
            "has_docstring": docstring is not None,
            "method_count": len(methods),
            "base_class_count": len(bases),
            "token_count": len(self.encoding.encode(content)),
        }

        return {"id": element_id, "content": content, "metadata": metadata}

    def _extract_function_info(
        self,
        python_file: Path,
        project_name: str,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source_code: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract function information.

        Args:
            python_file: Path to Python file
            project_name: Project name
            node: AST function node
            source_code: Source code string

        Returns:
            Function element dictionary or None
        """
        # Get function docstring
        docstring = ast.get_docstring(node)

        # Get function arguments
        args = []
        if node.args.args:
            for arg in node.args.args:
                args.append(arg.arg)

        # Check if it's a method (has 'self' or 'cls' as first arg)
        is_method = len(args) > 0 and args[0] in ["self", "cls"]
        element_type = "method" if is_method else "function"

        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)

        # Get relative path
        try:
            relative_path = python_file.relative_to(self.workspace_root)
        except ValueError:
            relative_path = python_file

        # Create content
        content_parts = []

        function_type = "Method" if is_method else "Function"
        content_parts.append(f"{function_type}: {node.name}")

        if args:
            content_parts.append(f"Arguments: ({', '.join(args)})")

        if decorators:
            content_parts.append(f"Decorators: @{', @'.join(decorators)}")

        if docstring:
            content_parts.append(f"Documentation: {docstring}")

        # Add some source code context (function signature and a few lines)
        lines = source_code.split("\n")
        if hasattr(node, "lineno") and node.lineno <= len(lines):
            start_line = max(0, node.lineno - 1)
            context_lines = []
            for i in range(start_line, min(len(lines), start_line + 3)):
                line = lines[i].strip()
                if line:
                    context_lines.append(line)

            if context_lines:
                content_parts.append(f"Definition: {' | '.join(context_lines)}")

        content = "\n\n".join(content_parts)

        rel_posix = relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path).replace(os.sep, "/")
        lineno = getattr(node, "lineno", 0)
        element_id = f"{project_name}__{element_type}__{rel_posix}__{node.name}@{lineno}"

        metadata = {
            "source": "codebase",
            "project": project_name,
            "file_path": str(rel_posix),
            "element_type": element_type,
            "element_name": node.name,
            "priority": self.element_priorities[element_type],
            "line_number": getattr(node, "lineno", 0),
            "has_docstring": docstring is not None,
            "argument_count": len(args),
            "decorator_count": len(decorators),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "token_count": len(self.encoding.encode(content)),
        }

        return {"id": element_id, "content": content, "metadata": metadata}

    def search_codebase(
        self,
        query: str,
        project: Optional[str] = None,
        element_type: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search codebase content.

        Args:
            query: Search query
            project: Optional project filter
            element_type: Optional element type filter (class, function, method, module)
            n_results: Number of results to return

        Returns:
            List of search results
        """
        where_filter = {}
        if project:
            where_filter["project"] = project
        if element_type:
            where_filter["element_type"] = element_type

        where_filter = where_filter or None

        return self.vector_store.search_documents(
            collection_name="codebase",
            query=query,
            n_results=n_results,
            where=where_filter,
        )

    def get_codebase_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed codebase.

        Returns:
            Dictionary with indexing statistics
        """
        try:
            collection_stats = self.vector_store.get_collection_stats()
            code_stats = collection_stats.get("codebase", {})

            return {
                "total_elements": code_stats.get("document_count", 0),
                "projects": self.sck_projects,
                "workspace_root": str(self.workspace_root),
                "element_types": list(self.element_priorities.keys()),
                "max_chunk_tokens": self.max_chunk_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
            }

        except Exception as e:
            logger.error(f"Error getting codebase stats: {e}")
            return {"error": str(e)}
