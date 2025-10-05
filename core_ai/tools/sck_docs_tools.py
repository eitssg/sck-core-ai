"""
Simple SCK Tools for Langflow
Direct integration with SCK components without MCP bridge complexity
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

DOCS_PATH = "D:/Development/simple-cloud-kit-oss/simple-cloud-kit/sck-core-docs/build"


class SCKDocumentationTool:
    """Simple tool to search SCK documentation directly."""

    @staticmethod
    def search_documentation(query: str) -> str:
        """Search SCK documentation using simple file search."""
        try:
            # Search in the SCK docs build directory (try multiple locations)
            possible_docs_paths = [Path(DOCS_PATH)]

            docs_path = None
            for path in possible_docs_paths:
                if path.exists():
                    docs_path = path
                    break

            if not docs_path:
                return f"Documentation not found. Tried paths: {[str(p) for p in possible_docs_paths]}"

            # Simple text search in HTML files
            results = []
            for html_file in docs_path.rglob("*.html"):
                try:
                    content = html_file.read_text(encoding="utf-8")
                    if query.lower() in content.lower():
                        # Extract a snippet around the match
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                snippet = "\n".join(lines[start:end])
                                results.append(f"Found in {html_file.name}:\n{snippet}\n")
                                break
                        if len(results) >= 3:  # Limit results
                            break
                except Exception as e:
                    continue

            if results:
                return f"SCK Documentation Search Results for '{query}':\n\n" + "\n---\n".join(results)
            else:
                return (
                    f"No documentation found for '{query}'. Try broader terms like 'architecture', 'lambda', 'S3', or 'framework'."
                )

        except Exception as e:
            return f"Error searching documentation: {str(e)}"
