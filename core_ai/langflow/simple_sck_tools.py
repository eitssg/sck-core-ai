"""
Simple SCK Tools for Langflow
Direct integration with SCK components without MCP bridge complexity
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class SCKArchitectureTool:
    """Tool to provide SCK architecture information."""

    @staticmethod
    def get_architecture_info(component: str = "") -> str:
        """Get SCK architecture information."""

        architecture_info = {
            "overview": """
SCK (Simple Cloud Kit) Architecture Overview:

🏗️ **Core Execution Chain:**
CLI/UI → core-invoker → core-runner → [core-deployspec, core-component]

📦 **Python Modules (17+ submodules):**
- sck-core-framework: Base utilities, configuration, data models
- sck-core-logging: Structured logging with correlation IDs  
- sck-core-db: Database interface, DynamoDB/PynamoDB helpers
- sck-core-execute: Lambda step-function execution engine
- sck-core-runner: Kicks off core_execute step functions
- sck-core-deployspec: Generates and compiles deployment specs
- sck-core-component: Manages components in S3 and DynamoDB
- sck-core-invoker: Orchestrates lambda function execution
- sck-core-organization: AWS Organizations, SCPs, Accounts
- sck-core-api: FastAPI dev server, AWS API Gateway prod
- sck-core-codecommit: AWS CodeCommit event handling
- sck-core-cli: Command-line interface

🗄️ **S3 Architecture:**
Three bucket prefixes with lifecycle management:
- packages/: Input files (Jinja2 templates, deployment packages)
- files/: Pre-compilation resources and post-compilation files
- artefacts/: Post-compilation CloudFormation templates

⚡ **Lambda Runtime Model:**
- All Python runs in AWS Lambda (synchronous handlers only)
- Use ProxyEvent(**event) for API Gateway integration
- NO async/await in Lambda code
- Use MagicS3Bucket for bucket operations, boto3 client for presigned URLs
            """,
            "imports": """
📋 **Standard SCK Import Patterns:**

```python
# Core imports (MANDATORY)
import core_framework as util
import core_logging as log
import core_helper.aws as aws
from core_framework import ProxyEvent

# Logging setup (CRITICAL)
logger = log  # NOT log.get_logger()

# S3 operations
from core_helper.magic import MagicS3Bucket
bucket = MagicS3Bucket(bucket_name=util.get_bucket_name(), region=util.get_bucket_region())
```
            """,
            "lambda": """
🚀 **Lambda Handler Pattern:**

```python
from core_framework import ProxyEvent
import core_logging as log
import json

logger = log

def lambda_handler(event, context):
    \"\"\"Standard SCK Lambda handler.\"\"\"
    request = ProxyEvent(**event)
    logger.info("Processing request", extra={"path": request.path})
    
    # Process request
    result = {"message": "Success"}
    
    # Return envelope response
    return {
        "statusCode": 200,
        "body": json.dumps({
            "status": "success",
            "code": 200,
            "data": result,
            "metadata": {},
            "message": "Request processed successfully"
        })
    }
```
            """,
            "s3": """
🗄️ **S3 Operations Pattern:**

```python
from core_helper.magic import MagicS3Bucket
import core_framework as util
import boto3

# For bucket operations (put, get, list)
bucket = MagicS3Bucket(
    bucket_name=util.get_bucket_name(), 
    region=util.get_bucket_region()
)
bucket.put_object(Key="packages/template.yaml", Body=content)

# For presigned URLs (MagicS3Bucket doesn't support this)
s3_client = boto3.client('s3', region_name=util.get_bucket_region())
presigned_url = s3_client.generate_presigned_url(
    'put_object',
    Params={'Bucket': util.get_bucket_name(), 'Key': 'files/upload.zip'},
    ExpiresIn=3600
)
```
            """,
        }

        if component.lower() in architecture_info:
            return architecture_info[component.lower()]
        elif component:
            return f"Component '{component}' not found. Available: {', '.join(architecture_info.keys())}"
        else:
            return architecture_info["overview"]


# Export tools for easy access
def get_sck_tools() -> dict[str, Any]:
    """Get all SCK tools."""
    return {
        "search_documentation": SCKDocumentationTool.search_documentation,
        "search_codebase": SCKCodeSearchTool.search_codebase,
        "get_architecture": SCKArchitectureTool.get_architecture_info,
    }


# Test the tools
if __name__ == "__main__":
    tools = get_sck_tools()

    print("\nTesting SCK Documentation Search...")
    result = tools["search_documentation"]("architecture")
    print(result[:200] + "..." if len(result) > 200 else result)

    print("\nTesting SCK Codebase Search...")
    result = tools["search_codebase"]("ProxyEvent")
    print(result[:200] + "..." if len(result) > 200 else result)

    print("\nTesting SCK Architecture Info...")
    result = tools["get_architecture"]("lambda")
    print(result)
