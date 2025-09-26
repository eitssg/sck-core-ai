# Copilot Instructions (Submodule: sck-core-ai)

- **Tech**: Python package (AI agent for YAML/CloudFormation processing)
- **Precedence**: Local first; fallback to root `../../.github/copilot-instructions.md`
- **Conventions**: Follow `../sck-core-ui/docs/backend-code-style.md` for AWS/Lambda patterns

## AI Agent Architecture & Patterns

### Core Design Principles
- **Langflow Integration**: All AI agent logic implemented as Langflow workflows for visual debugging and non-technical modification
- **Multi-Interface Support**: Same core logic exposed via AWS Lambda, FastAPI server, and MCP server interfaces
- **SCK Framework Compliance**: Use `core_framework`, `core_logging`, `core_helper.aws` for consistency with other modules
- **Async-Compatible**: Design for both synchronous Lambda handlers and async MCP server operations

### Lambda Handler Requirements
```python
# CORRECT: Synchronous Lambda handler with ProxyEvent
from core_framework import ProxyEvent
import core_logging as log

def lambda_handler(event, context):
    request = ProxyEvent(**event)
    # Process with Langflow agent
    result = langflow_client.process_sync(request.json)
    return {"statusCode": 200, "body": json.dumps(result)}
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
- **Client Pattern**: Use synchronous Langflow client for Lambda compatibility
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

### Local Development Setup
```bash
# CORRECT: Development environment setup
uv sync --dev                    # Install with dev dependencies
cp .env.example .env            # Configure environment
pre-commit install              # Set up code quality hooks
uv run pytest                   # Verify installation
```

### Testing Requirements
- **Unit Tests**: Individual agent functions and Langflow integrations
- **Integration Tests**: End-to-end API and MCP server flows
- **Langflow Tests**: Validate workflow execution and outputs
- **Performance Tests**: Response times for large CloudFormation templates

### Code Quality Standards
```python
# CORRECT: Type hints and documentation
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

async def lint_yaml_content(
    content: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Lint YAML content using AI agent.
    
    Args:
        content: Raw YAML string to validate
        options: Optional validation configuration
        
    Returns:
        Validation result with errors and suggestions
        
    Raises:
        ValidationError: If content cannot be processed
    """
    logger.info("Starting YAML validation", content_length=len(content))
    # Implementation here
```

## Contradiction Detection & Resolution

### Common Anti-Patterns to Flag
- **Async in Lambda**: Using `async def` for Lambda handlers conflicts with SCK synchronous pattern
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