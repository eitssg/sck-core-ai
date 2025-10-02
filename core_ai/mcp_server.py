"""
MCP (Model Context Protocol) server for SCK Core AI agent.

Provides MCP tool interfaces for AI assistants to interact with
YAML/CloudFormation linting and validation capabilities.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Sequence

import asyncio
import json
import os
import sys

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    # MCP protocol: No stdout output allowed - using stderr for debug
    print(f"Loaded environment from: {env_file}", file=sys.stderr)
else:
    load_dotenv()  # Load from current directory or environment
    print("Loaded environment from current directory or system environment", file=sys.stderr)

from bs4 import BeautifulSoup
from core_ai.tools.registry import AnalysisTypes, CodeElementType, ContextStrategy, DocNames, LintModes  # noqa: E402
from mcp.types import TextContent, ToolAnnotations  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

# Import SCK framework components (when available)
import core_logging as logger  # noqa: E402

from core_ai.langflow.client import LangflowClient  # noqa: E402

from core_ai.indexing import ContextManager  # noqa: E402
from core_ai.indexing import get_availability_status  # noqa: E402
import sys
import json
import asyncio
from typing import Any, List
from mcp.server.fastmcp import FastMCP
import os
from chromadb import Client
from sentence_transformers import SentenceTransformer


chroma_client = Client()  # Persistent: Client(Settings(persist_directory="./db"))
collection = chroma_client.create_collection("docs")
model = SentenceTransformer('all-MiniLM-L6-v2')


# Index local documentation (run once or on file changes)
def index_documentation(doc_dir: str = "../sck-core-docs/build"):
    """Index HTML, text, and markdown files in the specified directory."""
    documents = []
    doc_ids = []
    idx = 0
    for root, dirs, files in os.walk(doc_dir):
        for filename in files:
            if filename.endswith((".html", ".htm", ".txt", ".md")):  # Include HTML extensions
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Handle HTML files
                        if filename.endswith((".html", ".htm")):
                            # Parse HTML and extract clean text
                            soup = BeautifulSoup(content, "html.parser")
                            # Remove script and style tags
                            for script in soup(["script", "style"]):
                                script.decompose()
                            # Extract text from body or main content
                            text = soup.get_text(separator=" ", strip=True)
                            # Clean up excessive whitespace
                            content = " ".join(text.split())
                        # For .txt/.md, use raw content (already clean)
                        documents.append(content)
                        doc_ids.append(f"doc_{idx}")
                        idx += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
    if documents:
        embeddings = model.encode(documents).tolist()
        collection.add(documents=documents, embeddings=embeddings, ids=doc_ids)
        print(f"Indexed {len(documents)} documents.", file=sys.stderr)


def _process_exception_data(e: Exception | None, code=500, message: str = "Tool execution failed") -> Dict[str, Any]:
    rv = {
        "status": "error",
        "code": code,
    }
    rv["message"] = message if e is None else f"{message}: {str(e)}"
    return rv


def _process_exception(
    e: Exception | None = None, code: int = 500, message: str = "Tool execution failed"
) -> Sequence[TextContent]:
    return [
        TextContent(type="text", text=json.dumps(_process_exception_data(e, code, message), indent=2)),
    ]


def _process_success(code: int = 200, data: dict | None = None) -> Sequence[TextContent]:
    rv = {"status": "success", "code": code}
    if data is not None:
        rv["data"] = data
    return [TextContent(type="text", text=json.dumps(rv, indent=2))]


class SCKCoreAIMCPServer:
    """MCP server for SCK Core AI agent."""

    def __init__(self):
        """
        Initialize MCP server.

        Args:
            workspace_root: Path to workspace root (auto-detected if None)
            build_directory: Path to documentation build directory (auto-detected if None)
        """
        # Log startup and environment info
        logger.info("Initializing SCK Core AI MCP Server")
        logger.info(f"LOCAL_MODE: {os.getenv('LOCAL_MODE', 'Not set')}")
        logger.info(f"LOG_DIR: {os.getenv('LOG_DIR', 'Not set')}")
        logger.info(f"VOLUME: {os.getenv('VOLUME', 'Not set')}")
        logger.info(f"CLIENT: {os.getenv('CLIENT', 'Not set')}")
        logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'Not set')}")

        self.langflow_client: Optional[LangflowClient] = None

        # Initialize indexing system if available
        self.context_manager: Optional[ContextManager] = None

        try:
            # Auto-detect paths if not provided
            # Assume we're in sck-core-ai, go up to workspace root
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            build_directory = os.path.join(workspace_root, "sck-core-docs", "build")
            self.context_manager = ContextManager(build_directory=build_directory, workspace_root=workspace_root)
            logger.info(f"Indexing system initialized with workspace: {workspace_root}")

        except Exception as e:
            logger.warning(f"Failed to initialize indexing system: {e}")
            self.context_manager = None

    def _process_langflow_sync(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through Langflow synchronously."""
        if not self.langflow_client:
            # Initialize Langflow client if not already done
            self.langflow_client = LangflowClient()

        try:
            result = self.langflow_client.process_sync(request)
            return result
        except Exception as e:
            logger.error(f"Langflow processing failed: {e}")
            return _process_exception_data(e, message="AI processing failed")

    async def handle_lint_yaml(self, **kwargs) -> Sequence[TextContent]:
        """Handle YAML linting requests."""
        content = kwargs.get("content", "")
        mode = kwargs.get("mode", "syntax")

        # Process through Langflow in thread pool (sync to async)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_langflow_sync,
            {
                "input_value": content,
                "tweaks": {
                    "yaml-parser": {"validation_mode": mode},
                    "response-formatter": {"format_type": "mcp_response"},
                },
            },
        )

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_validate_cloudformation(self, **kwargs) -> Sequence[TextContent]:
        """Handle CloudFormation validation requests."""

        template = kwargs.get("template", "")
        region = kwargs.get("region", "us-east-1")
        strict = kwargs.get("strict", True)

        # Process through Langflow
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_langflow_sync,
            {
                "input_value": template,
                "tweaks": {
                    "cf-validator": {"region": region, "strict_mode": strict},
                    "response-formatter": {"format_type": "mcp_response"},
                },
            },
        )

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_suggest_completion(self, **kwargs) -> Sequence[TextContent]:
        """Handle code completion requests."""

        content = kwargs.get("content", "")
        cursor_line = kwargs.get("cursor_line", 1)
        cursor_column = kwargs.get("cursor_column", 1)
        context_type = kwargs.get("context_type", "auto")

        # Prepare AI prompt for completion
        completion_prompt = f"""Provide code completion suggestions for the following content.

Content type: {context_type}
Cursor position: Line {cursor_line}, Column {cursor_column}

Content:
{content}

Return JSON array of suggestions with: text, description, insertText, kind."""

        # Process through Langflow
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_langflow_sync,
            {
                "input_value": completion_prompt,
                "tweaks": {
                    "ai-analyzer": {
                        "system_message": "You are a YAML/CloudFormation completion assistant. Provide helpful, contextually relevant suggestions.",
                        "temperature": 0.1,
                    },
                    "response-formatter": {"format_type": "mcp_response"},
                },
            },
        )

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_analyze_template(self, **kwargs) -> Sequence[TextContent]:
        """Handle deep template analysis requests."""
        template = kwargs.get("template", "")
        analysis_type = kwargs.get("analysis_type", "comprehensive")
        region = kwargs.get("region", "us-east-1")

        # Prepare analysis prompt
        analysis_prompts = {
            "security": "Focus on security vulnerabilities, IAM policies, encryption, and access controls.",
            "cost": "Analyze cost optimization opportunities, resource sizing, and billing implications.",
            "best_practices": "Evaluate adherence to AWS Well-Architected Framework principles.",
            "comprehensive": "Perform complete analysis covering security, cost, performance, and best practices.",
        }

        system_message = f"""You are an expert AWS CloudFormation analyst. {analysis_prompts.get(analysis_type, analysis_prompts['comprehensive'])}

Analyze the provided CloudFormation template and provide detailed findings with:
1. Specific issues and recommendations
2. Risk levels (high/medium/low)
3. Implementation guidance
4. References to AWS documentation"""

        # Process through Langflow
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_langflow_sync,
            {
                "input_value": template,
                "tweaks": {
                    "ai-analyzer": {
                        "system_message": system_message,
                        "temperature": 0.1,
                    },
                    "cf-validator": {"region": region, "strict_mode": True},
                    "response-formatter": {"format_type": "mcp_response"},
                },
            },
        )

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_search_documentation(self, **kwargs) -> Sequence[TextContent]:
        """Handle documentation search requests."""
        if not self.context_manager:
            return _process_exception(code=503, message="Indexing system not available")

        query = kwargs.get("query", "")
        section = kwargs.get("section")
        max_results = kwargs.get("max_results", 5)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.context_manager.doc_indexer.search_documentation,
                query,
                section,
                max_results,
            )

            return _process_success(
                data={
                    "query": query,
                    "section": section,
                    "results": results,
                    "count": len(results),
                },
            )

        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
            return _process_exception(e, code=500, message="Documentation search failed")

    async def handle_search_codebase(self, **kwargs) -> Sequence[TextContent]:
        """Handle codebase search requests."""
        if not self.context_manager:
            return _process_exception(code=503, message="Indexing system not available")

        query = kwargs.get("query", "")
        project = kwargs.get("project")
        element_type = kwargs.get("element_type")
        max_results = kwargs.get("max_results", 5)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.context_manager.code_indexer.search_codebase,
                query,
                project,
                element_type,
                max_results,
            )

            return _process_success(
                data={
                    "query": query,
                    "project": project,
                    "element_type": element_type,
                    "results": results,
                    "count": len(results),
                },
            )

        except Exception as e:
            logger.error(f"Codebase search failed: {e}")
            return _process_exception(e, code=500, message="Codebase search failed")

    async def handle_get_context_for_query(self, **kwargs) -> Sequence[TextContent]:
        """Handle comprehensive context retrieval requests."""
        if not self.context_manager:
            return _process_exception(code=503, message="Indexing system not available")

        query = kwargs.get("query", "")
        strategy = kwargs.get("strategy", "balanced")
        max_results = kwargs.get("max_results", 10)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            context = await loop.run_in_executor(
                None,
                self.context_manager.get_context_for_query,
                query,
                strategy,
                max_results,
                True,
            )

            return _process_success(data=context)

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return _process_exception(e, code=500, message="Context retrieval failed")

    async def handle_initialize_indexes(self, **kwargs) -> Sequence[TextContent]:
        """Handle index initialization requests."""
        if not self.context_manager:
            return _process_exception(code=503, message="Indexing system not available")

        force_rebuild = kwargs.get("force_rebuild", False)

        try:
            # Run in thread pool as this can take a while
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.context_manager.initialize_indexes, force_rebuild)

            return _process_success(
                data={
                    "force_rebuild": force_rebuild,
                    "results": results,
                },
            )

        except Exception as e:
            logger.error(f"Index initialization failed: {e}")
            return _process_exception(e, code=500, message="Index initialization failed")

    async def handle_get_indexing_stats(self, **kwargs) -> Sequence[TextContent]:
        """Handle indexing statistics requests."""
        if not self.context_manager:
            return _process_exception(code=503, message="Indexing system not available")

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(None, self.context_manager.get_system_stats)

            return _process_success(data=stats)

        except Exception as e:
            logger.error(f"Getting indexing stats failed: {e}")
            return _process_exception(e, code=500, message="Getting indexing stats failed")


sck_tool: SCKCoreAIMCPServer


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle: index docs on startup."""

    global sck_tool

    # Startup: Index documentation
    sck_tool = SCKCoreAIMCPServer()

    # Index documentation on startup
    index_documentation(doc_dir="../sck-core-docs/build")  # Replace with your docs folder
    print("Documentation indexed successfully.", file=sys.stderr)

    yield  # Server runs here

    # Shutdown: Optional cleanup (e.g., close DB)
    print("Server shutdown; DB closed.", file=sys.stderr)


mcp = FastMCP(name="simple-cloud-kit-mcp", lifespan=lifespan)  # Name your server


@mcp.tool()
async def availability_tool() -> Dict[str, Any]:
    """Check availability of resources."""
    return get_availability_status()


@mcp.tool(
    name="lint_yaml",
    description="Lint and validate YAML content with AI-powered suggestions",
    title="YAML Lint Tool",
    annotations=ToolAnnotations(title="YAML Lint Tool"),
    structured_output=False,
)
async def lint_yaml_tool(content: str, mode: LintModes = LintModes.syntax, strict: bool = True) -> Sequence[TextContent]:
    """Lint YAML documents."""
    try:
        logger.info("Linting YAML content")
        result = await sck_tool.handle_lint_yaml(content=content, mode=mode, strict=strict)
    except Exception as e:
        logger.error(f"YAML linting tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="validate_cloudformation",
    description="Validate CloudFormation templates with comprehensive analysis",
    title="CloudFormation Lint Tool",
    annotations=ToolAnnotations(title="CloudFormation Lint Tool"),
    structured_output=False,
)
async def lint_cloudformation_tool(template: str, region: str = "us-east-1", strict: bool = True) -> Sequence[TextContent]:
    """Lint CloudFormation templates."""
    try:
        logger.info("Validating CloudFormation template")
        result = await sck_tool.handle_validate_cloudformation(template=template, region=region, strict=strict)
    except Exception as e:
        logger.error(f"CloudFormation linting tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="suggest_completion",
    description="Get AI-powered code completion suggestions for YAML/CloudFormation",
    title="Content Completion",
    annotations=ToolAnnotations(title="Suggestions for completions"),
    structured_output=False,
)
async def suggest_completion(
    content: str, cursor_line: int, cursor_column: int, context_type: str = "auto"
) -> Sequence[TextContent]:
    """Get code completion suggestions."""
    try:
        logger.info("Getting code completion suggestions")
        result = await sck_tool.handle_suggest_completion(
            content=content, cursor_line=cursor_line, cursor_column=cursor_column, context_type=context_type
        )
    except Exception as e:
        logger.error(f"Code completion suggestion tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="analyze_template",
    description="Perform deep analysis of CloudFormation templates for security, cost, and best practices",
    title="Analyze SCK Template",
    annotations=ToolAnnotations(title="Analyze Core SCK template"),
    structured_output=False,
)
async def analyze_template(
    template: str, analysis_type: AnalysisTypes = AnalysisTypes.comprehensive, region: str = "us-east-1"
) -> Sequence[TextContent]:
    """Analyze CloudFormation templates."""
    try:
        logger.info("Analyzing CloudFormation template")
        result = await sck_tool.handle_analyze_template(template=template, analysis_type=analysis_type, region=region)
    except Exception as e:
        logger.error(f"CloudFormation template analysis tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="search_documentation",
    description="Search SCK documentation",
    title="Search Documentation",
    annotations=ToolAnnotations(title="Search SCK documentation"),
    structured_output=False,
)
async def search_documentation(
    query: str, section: DocNames = DocNames.technical_reference, max_results: int = 5
) -> Sequence[TextContent]:
    """Search documentation."""
    try:
        logger.info("Searching documentation")
        result = await sck_tool.handle_search_documentation(query=query, section=section, max_results=max_results)
    except Exception as e:
        logger.error(f"Documentation search tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="search_codebase",
    description="Search SCK codebase",
    title="Search Codebase",
    annotations=ToolAnnotations(title="Search SCK codebase"),
    structured_output=False,
)
async def search_codebase(
    query: str, project: str = "", element_type: CodeElementType = CodeElementType.klass, max_results: int = 5
) -> Sequence[TextContent]:
    """Search codebase."""
    try:
        logger.info("Searching codebase")
        result = await sck_tool.handle_search_codebase(
            query=query, project=project, element_type=element_type, max_results=max_results
        )
    except Exception as e:
        logger.error(f"Codebase search tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="get_context_for_query",
    description="Get comprehensive context from both documentation and codebase for development assistance",
    title="Get Context for Query",
    annotations=ToolAnnotations(title="Get Context for Query"),
    structured_output=False,
)
async def get_context_for_query(query: str, strategy: ContextStrategy, max_results: int = 5) -> Sequence[TextContent]:
    """Get context for query."""
    try:
        logger.info("Getting context for query")
        result = await sck_tool.handle_get_context_for_query(query=query, strategy=strategy, max_results=max_results)
    except Exception as e:
        logger.error(f"Get context for query tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="initialize_indexes",
    description="Initialize or rebuild documentation and codebase indexes",
    title="Inizialize All Indexes",
    annotations=ToolAnnotations(title="Initialize Indexes"),
    structured_output=False,
)
async def initialize_indexes(force_rebuild: bool = False) -> Sequence[TextContent]:
    """Initialize or rebuild indexes."""
    try:
        logger.info("Initializing indexes")
        result = await sck_tool.handle_initialize_indexes(force_rebuild=force_rebuild)
    except Exception as e:
        logger.error(f"Initialize indexes tool failed: {e}")
        result = _process_exception(e)
    return result


@mcp.tool(
    name="get_indexing_stats",
    description="Get statistics about indexed documentation and codebase",
    structured_output=False,
)
async def get_indexing_status() -> Sequence[TextContent]:
    """Get indexing statistics."""
    try:
        logger.info("Getting indexing stats")
        result = await sck_tool.handle_get_indexing_stats()
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], TextContent):
            # Parse JSON from TextContent
            return result
        else:
            return _process_exception(message="Unexpected response format")
    except Exception as e:
        logger.error(f"Get indexing stats tool failed: {e}")
        return _process_exception(e)


if __name__ == "__main__":
    # Run the async main method
    mcp.run(transport="stdio")
