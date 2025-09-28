# SCK Core AI - Intelligent Development Assistant

An AI-powered development assistant that combines YAML/CloudFormation analysis with comprehensive documentation and codebase indexing. Features semantic search across SCK documentation and source code, plus AI-powered development assistance through multiple interfaces.

## 🚀 Features

### Core AI Capabilities
- **YAML Linting & Validation**: Advanced YAML syntax checking and best practices validation
- **CloudFormation Analysis**: Deep CloudFormation template analysis, resource validation, and policy checking
- **AI-Powered Code Completion**: Intelligent suggestions for YAML and CloudFormation resources

### Advanced Indexing & Search
- **Documentation Indexing**: Semantic search across built Sphinx documentation (Technical Reference, Developer Guide, User Guide)
- **Codebase Analysis**: Index and search Python source code across all SCK projects (functions, classes, modules)
- **Vector Database**: ChromaDB-powered semantic search with sentence transformers
- **Context-Aware Assistance**: RAG (Retrieval Augmented Generation) for development queries

### Multi-Interface Support
- **AWS Lambda + API Gateway**: Production-ready serverless deployment
- **FastAPI Server**: Local development and testing
- **MCP Server**: Integration with AI assistants (VS Code Copilot, Claude, etc.)
- **CLI Tools**: Command-line testing and administration

### Integration & Compatibility
- **Langflow Integration**: Visual workflow builder for AI agent logic
- **SCK Framework Compatible**: Native integration with Simple Cloud Kit ecosystem
- **Vector Store Persistence**: Persistent semantic indexes for fast startup

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│  Lambda Handler │───▶│  Langflow Agent │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │───▶│   MCP Server    │───▶│  AI Processing  │
│   (VS Code)     │    │   Interface     │    │   Pipeline      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Development Setup

1. **Clone and navigate to the project**:
   ```bash
   cd sck-core-ai
   ```

2. **Install dependencies (Hybrid Approach - RECOMMENDED)**:
   
   **⚠️ Important**: uv has limitations with local wheel installations. Use this proven approach:
   
   ```bash
   # Step 1: Build sck-core-framework wheel first
   cd ../sck-core-framework
   poetry build                    # Creates wheel in dist/
   
   # Step 2: Create and activate virtual environment manually
   cd ../sck-core-ai
   python -m venv .venv           # Create clean venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   # OR
   source .venv/bin/activate      # Linux/Mac
   
   # Step 3: Install local wheel with pip (RELIABLE)
   pip install ../sck-core-framework/dist/sck_core_framework-*.whl
   
   # Step 4: Install remaining dependencies
   pip install -e .
   pip install pytest black flake8 mypy  # Dev dependencies
   
   # Step 5: Verify installation
   python -c "import core_logging; print('Success!')"
   ```
   
   **Alternative (Pure pip approach)**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate     # Linux/Mac
   pip install ../sck-core-framework/dist/sck_core_framework-*.whl
   pip install -e .
   pip install -r requirements-dev.txt  # If available
   ```

3. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Production Deployment

The service is designed to be deployed as an AWS Lambda function. Deployment patterns align with the rest of the Simple Cloud Kit modules (see root repository docs).

## Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_LAMBDA_FUNCTION_NAME=sck-core-ai

# Langflow Configuration
LANGFLOW_HOST=localhost
LANGFLOW_PORT=7860
LANGFLOW_FLOW_ID=your-flow-id

# AI Model Configuration
OPENAI_API_KEY=your-openai-key  # Optional: for OpenAI models
ANTHROPIC_API_KEY=your-anthropic-key  # Optional: for Claude models

# Local Development
LOCAL_MODE=true
LOG_LEVEL=DEBUG

# Internal Idempotency (local service layer cache)
CORE_AI_INTERNAL_IDEMPOTENCY_ENABLED=true
CORE_AI_INTERNAL_IDEMPOTENCY_TTL=600  # seconds

# Vector Search (optional)
# Install extras: pip install scikit-learn sentence-transformers
# If not installed, search endpoints return empty results gracefully.
VECTOR_STORE_PERSIST_DIR=./data/vectordb
```

### Langflow Configuration

The AI agent logic is defined in Langflow workflows. See `langflow/` directory for:
- `ai-agent-flow.json`: Main agent workflow
- `templates/`: Reusable workflow components
- `README.md`: Langflow setup instructions

## Usage

### As Lambda Function (Production)

```bash
# Deploy using SAM or CDK
sam deploy --config-file samconfig.toml
```

### As Local FastAPI Server

```bash
# Start development server (PowerShell helper script)
./start.ps1  # defaults to port 8200

# Or start with uvicorn directly
uvicorn core_ai.server:app --reload --port 8200

# Or using uv (if installed)
uv run uvicorn core_ai.server:app --reload --port 8200
```

Port selection note:
> The default port was changed from 8000 to 8200 to avoid conflict with local DynamoDB Local which commonly binds to 8000. The precedence rules are: explicit `-Port` parameter to `start.ps1` > `SCK_AI_PORT` env var > legacy `SERVER_PORT` env var > script default (8200).

Test endpoints:
```bash
# Health check
curl http://localhost:8200/health

# Lint YAML
curl -X POST http://localhost:8200/api/v1/lint/yaml \
  -H "Content-Type: application/json" \
  -d '{"content": "key: value\n  invalid: indentation"}'

# Validate CloudFormation
curl -X POST http://localhost:8200/api/v1/validate/cloudformation \
  -H "Content-Type: application/json" \
  -d @examples/template.json
```

### As MCP Server (Enhanced)

The MCP server now includes comprehensive documentation and codebase indexing:

```bash
# Test the indexing system
python test_indexing.py

# Start MCP server with indexing
python run_mcp_server.py --initialize-indexes

# Or start without initializing (faster startup)
python run_mcp_server.py
```

**MCP Tools Available:**

1. **YAML/CloudFormation Tools** (original):
   - `lint_yaml`: YAML linting and validation
   - `validate_cloudformation`: CloudFormation template validation
   - `suggest_completion`: AI-powered code completion
   - `analyze_template`: Deep template analysis

2. **Documentation & Codebase Tools** (new):
   - `search_documentation`: Search SCK documentation
   - `search_codebase`: Search Python source code
   - `get_context_for_query`: Comprehensive development context
   - `initialize_indexes`: Build/rebuild search indexes
   - `get_indexing_stats`: Index statistics and health

**Configure in VS Code (`settings.json`):**
```json
{
  "mcp.servers": {
    "sck-core-ai": {
      "command": "python",
      "args": ["run_mcp_server.py"],
      "cwd": "/path/to/sck-core-ai",
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## API Reference

### REST API Endpoints

#### `POST /api/v1/lint/yaml`
Lint and validate YAML content.

**Request:**
```json
{
  "content": "yaml content here",
  "options": {
    "strict": true,
    "schema": "cloudformation"  // optional
  }
}
```

**Response:**
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "valid": false,
    "errors": [
      {
        "line": 2,
        "column": 1,
        "message": "Invalid indentation",
        "severity": "error"
      }
    ],
    "suggestions": [
      {
        "line": 2,
        "suggestion": "Fix indentation to match YAML standards"
      }
    ]
  }
}
```

#### `POST /api/v1/validate/cloudformation`
Validate CloudFormation templates.

#### `POST /api/v1/complete`
Get AI-powered code completion suggestions.

### MCP Tools

- `lint_yaml`: Lint YAML content
- `validate_cloudformation`: Validate CF templates  
- `suggest_completion`: Get completion suggestions
- `analyze_template`: Perform deep template analysis

## Development

### Project Structure

```
sck-core-ai/
├── core_ai/
│   ├── __init__.py
│   ├── server.py           # FastAPI/Lambda handler
│   ├── mcp_server.py       # MCP server implementation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── yaml_linter.py
│   │   └── cf_validator.py
│   ├── langflow/
│   │   ├── __init__.py
│   │   └── client.py       # Langflow integration
│   └── utils/
├── langflow/
│   ├── ai-agent-flow.json  # Main Langflow workflow
│   └── templates/
├── tests/
├── docs/
└── examples/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=core_ai --cov-report=html

# Run specific test file
uv run pytest tests/test_yaml_linter.py -v
```

### Code Quality

```bash
# Format code
uv run black .
uv run isort .

# Lint code
uv run flake8 core_ai/
uv run mypy core_ai/

# Run all quality checks
uv run pre-commit run --all-files
```

## Integration with SCK Ecosystem

This AI agent integrates with other Simple Cloud Kit modules:

- **core-framework**: Configuration and utilities
- **core-logging**: Structured logging
- **core-api**: API patterns and middleware
- **core-helper**: AWS service helpers

See the root Simple Cloud Kit documentation for broader integration details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the root repository LICENSE for details.

## Support

- **Documentation**: Refer to root repository docs and inline docstrings
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **SCK Community**: [Simple Cloud Kit Documentation](https://github.com/eitssg/simple-cloud-kit)
# sck-core-ai
