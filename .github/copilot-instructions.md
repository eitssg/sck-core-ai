# Copilot Instructions (Submodule: sck-core-ai)

## Plan → Approval → Execute (Mandatory – Overrides Prior Proactive Rules)
ALL non-trivial steps (file edits, Docker changes, Langflow workflow additions, test modifications) require an upfront plan and explicit approval. Trivial Q&A or explicit "skip plan" instructions are the only exceptions.

- **Tech**: Python package (AI agent for YAML/CloudFormation processing)
- **Precedence**: Local first; fallback to root `../../.github/copilot-instructions.md`
- **Conventions**: Follow `../sck-core-ui/docs/backend-code-style.md` for AWS + container service patterns (this module is NOT deployed as Lambda)

## Backward Compatibility (Explicitly Forbidden)
- Do **not** implement fallback, legacy, or backward-compatibility branches in this submodule.
- Prefer removing obsolete code paths instead of guarding them with feature flags.
- When behavior changes, update dependent callers/tests rather than adding shims.

## 🧪 UV Command Formulation (Mandatory)

When proposing or executing Python-related commands in this module:
- pip → `uv pip <args>`
- python → `uv python <args>`
- python -m <module> → `uv run -m <module> <args>`
- CLI tools (pytest/black/flake8/mypy/etc.) → `uv run <tool> <args>`

This complements the terminal protocol above: use the existing terminal, do not activate environments; uv resolves the environment.

## 🔧 CORE_LOGGING USAGE PROTOCOL 🔧

**RULE_101_LOGGING_IMPORT**: Always import as `import core_logging as log`

**CORRELATION_ID**: Use `log.set_correlation_id("value")` at start of request/operation to set context. This is thread_local and works in async/multi-threaded code.

**JSON_LOGGING**: Logs by default are strings, to use JSON, set the environment variable LOG_AS_JSON=true.  The variable is inspected at each new handler creation, so it can be toggled when you add a new identity.  Best to assume this is handled at boot time.

**RULE_102_LOGGER_ASSIGNMENT**: Use `import core_logging as log` (NOT `log.get_logger()`)

**RULE_103_LOGGING_CALLS**: Use `log.info()`, `log.error()`, `log.debug()` directly

**RULE_104_NO_GET_LOGGER**: NEVER call `log.get_logger(__name__)` - core_logging is pre-configured and automatically uses module name and correlation IDs.  To set a correlation ID, use `log.set_correlation_id("value")` at the start of a request or operation.  This is a thread_local context, so it works in async code and multi-threaded environments.

**RULE_105_IMPORTS_AT_TOP**: ALL imports must be at top of file, not in functions

## 🚨 PYTHON APPLICATION STRUCTURE PROTOCOL 🚨

**CRITICAL WARNING**: Claude Sonnet 3.5 does NOT understand proper Python application architecture and module structure.

**RULE_201_NO_PYTHON_STRUCTURE_CHANGES**: NEVER create new modules, packages, or application files without explicit developer approval

**RULE_202_NO_CONDITIONAL_IMPORTS**: NEVER use try/except ImportError patterns - imports are not optional and failures must be fixed, not masked

**RULE_203_NO_SYS_PATH_HACKS**: NEVER manipulate sys.path or use hacky import workarounds

**RULE_204_ASK_FOR_STRUCTURE**: When needing new Python files or modules, ASK the developer how they want the structure organized

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

### MCP Server Patterns
- **Tool Exposure**: Each agent function exposed as an MCP tool with clear type annotations
- **Input Validation**: Validate and sanitize all inputs at the MCP tool boundary
- **Error Handling**: Convert exceptions to structured SCK error responses
- **Logging**: Use `core_logging` with `import core_logging as log` for structured logs with correlation IDs.  On entry point, set correlation ID from request context with `log.set_correlation_id(...)`
- **FastMCP**: Utilize the FastMCP server framework for efficient request handling and concurrency

### Langflow Integration Standards
- **Flow Management**: Store flows in `langflow/` directory as JSON exports
- **Client Pattern**: Use synchronous Langflow client in main execution path; wrap heavy or multi-call chains in thread pool if latency spikes.
- **Error Handling**: Wrap Langflow calls with structured logging and error conversion
- **Configuration**: Flow IDs and endpoints via environment variables

## YAML/CloudFormation Processing Rules
### YAML Parsing Policy (Local to sck-core-ai)
- Use core_framework for YAML parsing in ALL modules. Example:
    - `import core_framework as util`, `data = util.read_yaml(stream)`, `util.write_yaml(data, stream)`, `data = util.from_yaml(string)`, `string = util.to_yaml(data)`
- Do not use try/except around imports; add dependencies instead.
 - Prefer core_framework YAML helpers where available: `from_yaml`, `to_yaml`, `read_yaml`, `write_yaml`, `load_yaml_file`. These incorporate custom constructors/tags and !Include handling.


### Validation Hierarchy
1. **Syntax Check**: Basic YAML/JSON parsing
2. **Schema Validation**: CloudFormation resource schema compliance  
3. **Best Practices**: AWS Well-Architected principles
4. **Security Analysis**: IAM policies, security groups, encryption
5. **Cost Optimization**: Resource sizing recommendations

### Code Completion Patterns
- **Context-Aware**: Parse existing template structure for relevant suggestions
- **Resource-Specific**: Tailor completions based on CloudFormation resource types
- **Best Practices**: Include security and cost optimization in suggestions
- **Multi-Format**: Support both YAML and JSON CloudFormation templates

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
  - Examples and code in docstring should include a single >>> prompt line, followed by indented code lines and is terminated with a blank line.
  ```python
  def code():
    """ Example function.

    Example:
    >>> # Code comment
        code()  
        42
        .
        sample = 43
        assert sample == 43
    
    Returns:
        int: The answer to life, the universe, and everything.
    """
    return 42
  ```

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

## Standalone Clone Note

If cloned standalone, see:
- UI/backend conventions: https://github.com/eitssg/simple-cloud-kit/tree/develop/sck-core-ui/docs
- Root Copilot guidance: https://github.com/eitssg/simple-cloud-kit/blob/develop/.github/copilot-instructions.md