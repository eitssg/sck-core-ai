"""
Simple SCK Tools for Langflow
Direct integration with SCK components without MCP bridge complexity
"""

from typing import Any, Dict, List, Optional

from pathlib import Path


class SCKCodeSearchTool:
    """Simple tool to search SCK codebase directly."""

    @staticmethod
    def search_codebase(query: str) -> str:
        """Search SCK codebase for patterns."""
        try:
            # Search in the SCK workspace
            workspace_path = Path("D:/Development/simple-cloud-kit-oss/simple-cloud-kit")

            if not workspace_path.exists():
                return f"Workspace not found at {workspace_path}"

            # Simple grep-like search in Python files
            results = []
            search_patterns = [
                "*.py",  # Python files
                "*.yaml",
                "*.yml",  # YAML files
                "*.json",  # JSON files
            ]

            for pattern in search_patterns:
                for py_file in workspace_path.rglob(pattern):
                    # Skip certain directories
                    if any(skip in str(py_file) for skip in [".venv", "node_modules", "__pycache__", ".git"]):
                        continue

                    try:
                        content = py_file.read_text(encoding="utf-8")
                        if query.lower() in content.lower():
                            # Extract relevant lines
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if query.lower() in line.lower():
                                    start = max(0, i - 1)
                                    end = min(len(lines), i + 2)
                                    snippet = "\n".join([f"{start+j+1}: {lines[start+j]}" for j in range(end - start)])
                                    results.append(f"Found in {py_file.relative_to(workspace_path)}:\n{snippet}\n")
                                    break
                            if len(results) >= 5:  # Limit results
                                break
                    except Exception as e:
                        continue

                if results:
                    break

            if results:
                return f"SCK Codebase Search Results for '{query}':\n\n" + "\n---\n".join(results)
            else:
                return f"No code found for '{query}'. Try terms like 'ProxyEvent', 'core_logging', 'MagicS3Bucket', or 'lambda_handler'."

        except Exception as e:
            return f"Error searching codebase: {str(e)}"
