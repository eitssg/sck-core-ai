# Copilot Instructions (Submodule: sck-core-ai)

- **Tech**: Python package (AI agent for YAML/CloudFormation processing)
- **Precedence**: Local first; fallback to root `../../.github/copilot-instructions.md`
- **Conventions**: Follow `../sck-core-ui/docs/backend-code-style.md` for AWS + container service patterns (this module is NOT deployed as Lambda)

## 🤖 LLM TERMINAL EXECUTION PROTOCOL 🤖

**ABSOLUTE RULES - NO EXCEPTIONS:**

**RULE_001_USE_ACTIVE_TERMINAL**: ALWAYS use the currently active terminal - it already has the correct environment

**RULE_002_VERIFY_WITH_LAST_COMMAND**: Use `terminal_last_command` to confirm terminal is in correct project directory

**RULE_003_NO_ENVIRONMENT_MANAGEMENT**: NEVER try to activate virtual environments - user has already configured each terminal

**RULE_004_TRUST_EXISTING_SETUP**: If terminal shows correct directory, trust that environment is correct

**RULE_005_MODULE_NOT_FOUND_MEANS_WRONG_TERMINAL**: If "ModuleNotFoundError", ask user to switch to correct terminal instead of trying to fix environment

**ENFORCEMENT**: Use existing configured terminals. Stop trying to manage environments.

## 🔧 CORE_LOGGING USAGE PROTOCOL 🔧

**RULE_101_LOGGING_IMPORT**: Always import as `import core_logging as log`

**RULE_102_LOGGER_ASSIGNMENT**: Set `logger = log` (NOT `log.get_logger()`)

**RULE_103_LOGGING_CALLS**: Use `logger.info()`, `logger.error()`, `logger.debug()` directly

**RULE_104_NO_GET_LOGGER**: NEVER call `log.get_logger(__name__)` - core_logging is pre-configured

**RULE_105_IMPORTS_AT_TOP**: ALL imports must be at top of file, not in functions

## 🚨 PYTHON APPLICATION STRUCTURE PROTOCOL 🚨

**CRITICAL WARNING**: Claude Sonnet 3.5 does NOT understand proper Python application architecture and module structure.

**RULE_201_NO_PYTHON_STRUCTURE_CHANGES**: NEVER create new modules, packages, or application files without explicit developer approval

**RULE_202_NO_CONDITIONAL_IMPORTS**: NEVER use try/except ImportError patterns - imports are not optional and failures must be fixed, not masked

**RULE_203_NO_SYS_PATH_HACKS**: NEVER manipulate sys.path or use hacky import workarounds

**RULE_204_ASK_FOR_STRUCTURE**: When needing new Python files or modules, ASK the developer how they want the structure organized

**RULE_205_INDIVIDUAL_FUNCTIONS_ONLY**: Claude can generate good individual Python functions and classes, but cannot architect proper module structure

**EXAMPLES OF FORBIDDEN PATTERNS**:
```python
# ❌ WRONG - Conditional imports
try:
    from some_module import SomeClass
except ImportError:
    class SomeClass: pass  # Fallback

# ❌ WRONG - sys.path manipulation  
sys.path.insert(0, "some/directory")
from local_module import something

# ❌ WRONG - Creating new application files without asking
```

**CORRECT APPROACH**: "How should I structure the imports for these tools? Should they be in a separate package, or how do you want the module organized?"

## AI Agent Architecture & Patterns

### Core Design Principles
- **Langflow Integration**: All AI agent logic implemented as Langflow workflows for visual debugging and non-technical modification
- **Multi-Interface Support**: Core logic exposed via containerized FastAPI app & MCP server (legacy Lambda example deprecated)
- **SCK Framework Compliance**: Use `core_framework`, `core_logging`, `core_helper.aws` for consistency with other modules
- **Async/Sync Strategy**: Prefer synchronous core execution for determinism; async allowed for I/O parallelism (MCP, external API calls)

### Container Service Entry Point (FastAPI)
```python
from fastapi import FastAPI
import core_logging as log

app = FastAPI()

@app.post("/ai/generate")
def generate(payload: dict):
    """Generate AI content using Langflow flow execution (sync core path)."""
    result = langflow_client.process_sync(payload)
    return {"status": "success", "data": result}
```

### MCP Server Patterns
```python
# CORRECT: Async MCP server methods
from mcp import types
import asyncio

async def handle_lint_yaml(arguments: dict) -> types.TextContent:
    # Convert to sync Langflow call in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, langflow_client.process_sync, arguments)
    return types.TextContent(text=json.dumps(result))
```

### Langflow Integration Standards
- **Flow Management**: Store flows in `langflow/` directory as JSON exports
- **Client Pattern**: Use synchronous Langflow client in main execution path; wrap heavy or multi-call chains in thread pool if latency spikes.
- **Error Handling**: Wrap Langflow calls with structured logging and error conversion
- **Configuration**: Flow IDs and endpoints via environment variables

## YAML/CloudFormation Processing Rules

### Validation Hierarchy
1. **Syntax Check**: Basic YAML/JSON parsing
2. **Schema Validation**: CloudFormation resource schema compliance  
3. **Best Practices**: AWS Well-Architected principles
4. **Security Analysis**: IAM policies, security groups, encryption
5. **Cost Optimization**: Resource sizing recommendations

### Response Envelope Standards
```python
# CORRECT: Structured validation response
{
    "status": "success",
    "code": 200, 
    "data": {
        "valid": False,
        "errors": [
            {
                "line": 42,
                "column": 10,
                "severity": "error",
                "code": "CF001",
                "message": "Invalid resource type",
                "suggestion": "Use AWS::EC2::Instance instead"
            }
        ],
        "warnings": [...],
        "suggestions": [...],
        "metrics": {
            "processing_time_ms": 150,
            "langflow_execution_id": "uuid"
        }
    }
}
```

### Code Completion Patterns
- **Context-Aware**: Parse existing template structure for relevant suggestions
- **Resource-Specific**: Tailor completions based on CloudFormation resource types
- **Best Practices**: Include security and cost optimization in suggestions
- **Multi-Format**: Support both YAML and JSON CloudFormation templates

## Development Workflow Standards

{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}### Local Development Setup (Hybrid Approach)

**CRITICAL**: uv has limitations with local wheel installations. Use this proven hybrid approach:

```bash
# Step 1: Build sck-core-framework wheel first
cd ../sck-core-framework
uv build                    # Creates wheel in dist/

# Step 2: Create and activate virtual environment manually
cd ../sck-core-ai
python -m venv .venv           # Create clean venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# OR
source .venv/bin/activate      # Linux/Mac

# Step 3: Install local wheel with pip (RELIABLE)
pip install ../sck-core-framework/dist/sck_core_framework-*.whl

# Step 4: Install remaining dependencies with uv (FAST)
uv pip install -e .            # Install current project in editable mode
uv pip install pytest black flake8 mypy  # Dev dependencies

# Step 5: Verify installation
python -c "import core_logging; print('Success!')"
pytest                         # Run tests
```

### Why This Hybrid Approach?

- **Manual venv**: Full control, no hidden uv magic
- **pip for local wheels**: Mature, reliable installation
- **uv for PyPI packages**: Fast, modern dependency resolution
- **Predictable**: Works every time, no surprises

### Alternative: Pure Traditional Approach
```bash
# If uv continues to cause issues, use pure pip:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ../sck-core-framework/dist/sck_core_framework-*.whl
pip install -e .
pip install -r requirements-dev.txt  # If you create this file
```

### Testing Requirements
- **Unit Tests**: Individual agent functions and Langflow integrations
- **Integration Tests**: End-to-end API and MCP server flows
- **Langflow Tests**: Validate workflow execution and outputs
- **Performance Tests**: Response times for large CloudFormation templates

### Google Docstring Requirements
**MANDATORY**: All docstrings must use Google-style format for Sphinx documentation generation:
- Use Google-style docstrings with proper Args/Returns/Example sections
- Napoleon extension will convert Google format to RST for Sphinx processing
- Avoid direct RST syntax (`::`, `:param:`, etc.) in docstrings - use Google format instead
- Example sections should use `>>>` for doctests or simple code examples
- This ensures proper IDE interpretation while maintaining clean Sphinx documentation

## Code Quality Standards
Example of RST-compatible docstring::

    from typing import Dict, List, Optional
    import structlog

    logger = structlog.get_logger(__name__)

    async def lint_yaml_content(
        content: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Lint YAML content using AI agent.
        
        :param content: Raw YAML string to validate
        :param options: Optional validation configuration
        :returns: Validation result with errors and suggestions
        :raises ValidationError: If content cannot be processed
        
        Example::
        
            logger.info("Starting YAML validation", content_length=len(content))
            # Implementation here
        """

## Contradiction Detection & Resolution

### Common Anti-Patterns to Flag
- **Container Anti-Pattern**: Re-introducing Lambda-specific lifecycle hacks (e.g., assuming cold start) or per-request reinitialization of heavy models.
- **Direct boto3**: Bypassing `core_helper.aws` conflicts with framework standards
- **Inline Langflow**: Embedding workflow logic in Python instead of using Langflow JSON files
- **Non-Envelope Responses**: Returning raw data instead of SCK API envelope format

### Langflow-Specific Rules
- **Flow Storage**: All workflows must be version-controlled as JSON exports in `langflow/` directory
- **Environment Isolation**: Separate flows for development, staging, and production environments  
- **Dependency Management**: Pin Langflow and LangChain versions for reproducible deployments
- **Error Propagation**: Convert Langflow exceptions to structured SCK error responses

### Integration Compliance
- **MCP Tools**: All agent functions must be exposed as MCP tools with proper type definitions
- **API Consistency**: REST endpoints must follow SCK envelope format from `../sck-core-ui/docs/backend-code-style.md`
- **Authentication**: When deployed, respect SCK auth patterns for protected endpoints
- **Monitoring**: Use `core_logging` for structured logs compatible with SCK observability

## Example Implementations

### Langflow Workflow Structure
```json
{
  "flow_id": "yaml-linter-v1",
  "nodes": [
    {
      "id": "input",
      "type": "TextInput", 
      "data": {"input_key": "yaml_content"}
    },
    {
      "id": "parser",
      "type": "YAMLParser",
      "inputs": {"text": "input.output"}
    },
    {
      "id": "validator", 
      "type": "CloudFormationValidator",
      "inputs": {"parsed_yaml": "parser.output"}
    },
    {
      "id": "ai_suggestions",
      "type": "OpenAIChat",
      "inputs": {"context": "validator.errors"}
    }
  ]
}
```

### MCP Server Tool Definition
```python
@mcp_server.tool()
async def lint_cloudformation_template(
    template: str,
    region: str = "us-east-1",
    strict: bool = True
) -> str:
    """
    Validate CloudFormation template with AI-powered suggestions.
    
    Args:
        template: CloudFormation template (YAML or JSON)
        region: AWS region for validation context
        strict: Enable strict validation rules
    """
    # Implementation using Langflow
```

## Standalone Clone Note

If cloned standalone, see:
- UI/backend conventions: https://github.com/eitssg/simple-cloud-kit/tree/develop/sck-core-ui/docs
- Root Copilot guidance: https://github.com/eitssg/simple-cloud-kit/blob/develop/.github/copilot-instructions.md