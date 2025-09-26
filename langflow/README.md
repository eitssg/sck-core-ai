# Langflow Configuration for SCK Core AI

This directory contains Langflow workflows and configuration files for the SCK Core AI agent.

## Workflow Files

### `ai-agent-flow.json`
Main AI agent workflow that processes YAML and CloudFormation templates through the following pipeline:

1. **Content Input**: Receives raw YAML/CloudFormation content
2. **YAML Parser**: Parses and validates YAML syntax
3. **CloudFormation Validator**: Validates CF-specific schemas and best practices  
4. **AI Analyzer**: Uses LLM for intelligent analysis and suggestions
5. **Response Formatter**: Formats output according to SCK API envelope standards

## Setup Instructions

### 1. Install Langflow

```bash
# Install Langflow with development dependencies
uv add "langflow[dev]>=1.0.0"

# Or using pip
pip install "langflow[dev]>=1.0.0"
```

### 2. Start Langflow Server

```bash
# Start Langflow development server
langflow run --host 0.0.0.0 --port 7860

# With custom configuration
langflow run --config langflow.conf
```

### 3. Import Workflow

1. Open Langflow UI at `http://localhost:7860`
2. Click "New Project" 
3. Select "Import from JSON"
4. Upload `ai-agent-flow.json`
5. Configure your API keys in the AI Analyzer node
6. Save and deploy the flow

### 4. Configuration

Set these environment variables for the AI agent:

```bash
# Langflow connection
LANGFLOW_HOST=localhost
LANGFLOW_PORT=7860
LANGFLOW_FLOW_ID=yaml-cf-ai-agent-v1

# AI model credentials (choose one or more)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint
AZURE_OPENAI_API_KEY=your-azure-key

# Regional settings for CloudFormation validation
AWS_DEFAULT_REGION=us-east-1
```

## Custom Components

The workflow uses several custom components that extend Langflow's base functionality:

### YAMLProcessor
- **Purpose**: Parse YAML with syntax validation and structure analysis
- **Inputs**: Raw YAML string, validation mode
- **Outputs**: Parsed YAML object, syntax errors, structure metadata
- **Implementation**: `core_ai/langflow/components/yaml_processor.py`

### CloudFormationValidator  
- **Purpose**: Validate CloudFormation templates against AWS schemas
- **Inputs**: Parsed template, AWS region, strict mode flag
- **Outputs**: Validation results, resource analysis, compliance report
- **Implementation**: `core_ai/langflow/components/cf_validator.py`

### ResponseFormatter
- **Purpose**: Format AI agent responses according to SCK API standards
- **Inputs**: Validation results, AI analysis, format type
- **Outputs**: Structured response envelope
- **Implementation**: `core_ai/langflow/components/response_formatter.py`

## Workflow Variations

### `templates/yaml-linter-simple.json`
Simplified workflow for basic YAML linting without AI analysis.

### `templates/cf-security-analyzer.json`  
Specialized workflow focused on CloudFormation security analysis.

### `templates/code-completion.json`
Workflow optimized for providing intelligent code completion suggestions.

## Integration with SCK Core AI

The Python application integrates with Langflow through:

```python
from core_ai.langflow.client import LangflowClient

# Initialize client
client = LangflowClient(
    base_url="http://localhost:7860",
    flow_id="yaml-cf-ai-agent-v1"
)

# Process content
result = client.run_flow(
    input_value="your yaml content here",
    tweaks={
        "ai-analyzer": {"model_name": "gpt-4"},
        "cf-validator": {"strict_mode": True}
    }
)
```

## Development Tips

### Testing Workflows
1. Use the Langflow playground to test individual components
2. Create test cases with known good/bad YAML files
3. Validate AI responses match expected SCK envelope format

### Debugging
- Enable debug logging in Langflow: `LANGFLOW_LOG_LEVEL=DEBUG`
- Check component logs in the Langflow UI
- Use the "View Logs" feature for each node

### Performance Optimization
- Cache AI model responses for similar inputs
- Use streaming for large CloudFormation templates
- Implement request deduplication for repeated validations

## Deployment

For production deployment, see:
- [Langflow Cloud deployment guide](https://docs.langflow.org/deployment)
- [Self-hosted Langflow with Docker](../docs/langflow-deployment.md)
- [AWS Lambda integration](../docs/lambda-integration.md)

## Troubleshooting

### Common Issues

**Flow not loading**: Check JSON syntax and node IDs are unique
**AI responses empty**: Verify API keys and model availability  
**Validation errors**: Ensure cfn-lint is properly installed and updated
**Performance issues**: Consider using smaller models or caching for development

### Getting Help

- [Langflow Documentation](https://docs.langflow.org/)
- [SCK Core AI Issues](https://github.com/eitssg/simple-cloud-kit/issues)
- [Community Discussions](https://github.com/eitssg/simple-cloud-kit/discussions)