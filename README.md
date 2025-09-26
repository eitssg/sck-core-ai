# SCK Core AI - Intelligent YAML/CloudFormation Agent

An AI-powered agent built with Langflow for YAML and CloudFormation template linting, validation, and code completion. This service runs as an AWS Lambda function behind API Gateway and can also serve as a Model Context Protocol (MCP) server.

## Features

- **YAML Linting & Validation**: Advanced YAML syntax checking and best practices validation
- **CloudFormation Analysis**: Deep CloudFormation template analysis, resource validation, and policy checking
- **AI-Powered Code Completion**: Intelligent suggestions for YAML and CloudFormation resources
- **Multi-Interface Support**: 
  - AWS Lambda + API Gateway for production
  - FastAPI server for local development
  - MCP server for integration with AI assistants
- **Langflow Integration**: Visual workflow builder for AI agent logic
- **SCK Framework Compatible**: Integrates seamlessly with Simple Cloud Kit ecosystem

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

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync --dev
   
   # Or using pip
   pip install -e .[dev]
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

The service is designed to be deployed as an AWS Lambda function. See [deployment documentation](docs/deployment.md) for details.

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
# Start development server
uv run sck-ai-server

# Or with uvicorn directly
uvicorn core_ai.server:app --reload --port 8000
```

Test endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Lint YAML
curl -X POST http://localhost:8000/api/v1/lint/yaml \
  -H "Content-Type: application/json" \
  -d '{"content": "key: value\n  invalid: indentation"}'

# Validate CloudFormation
curl -X POST http://localhost:8000/api/v1/validate/cloudformation \
  -H "Content-Type: application/json" \
  -d @examples/template.json
```

### As MCP Server

```bash
# Start MCP server
uv run sck-ai-mcp --port 3000

# Configure in your MCP client (VS Code, etc.)
{
  "mcpServers": {
    "sck-ai": {
      "command": "uv",
      "args": ["run", "sck-ai-mcp"],
      "cwd": "/path/to/sck-core-ai"
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

See [SCK Integration Guide](docs/sck-integration.md) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **SCK Community**: [Simple Cloud Kit Documentation](https://github.com/eitssg/simple-cloud-kit)
# sck-core-ai
