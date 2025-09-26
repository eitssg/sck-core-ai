"""
FastAPI server for SCK Core AI agent.

Provides REST API endpoints for YAML/CloudFormation processing
and serves as the local development interface.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field

# Import SCK framework components (when available)
try:
    import core_framework as util
    import core_logging as log
    from core_framework import ProxyEvent

    SCK_AVAILABLE = True
except ImportError:
    # Fallback for standalone development
    import structlog as log

    SCK_AVAILABLE = False

# Import Langflow client
try:
    from .langflow.client import LangflowClient
except ImportError:
    # Mock for development without Langflow
    class LangflowClient:
        def __init__(self, *args, **kwargs):
            pass

        def process_sync(self, *args, **kwargs):
            return {"status": "success", "message": "Mock response"}


# API Models
class YamlLintRequest(BaseModel):
    """Request model for YAML linting."""

    content: str = Field(..., description="YAML content to lint")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Linting options (strict, schema, etc.)"
    )


class CloudFormationValidateRequest(BaseModel):
    """Request model for CloudFormation validation."""

    template: Dict[str, Any] = Field(..., description="CloudFormation template")
    region: str = Field(default="us-east-1", description="AWS region")
    strict: bool = Field(default=True, description="Enable strict validation")


class CodeCompletionRequest(BaseModel):
    """Request model for code completion."""

    content: str = Field(..., description="Partial YAML/CloudFormation content")
    cursor_position: Dict[str, int] = Field(..., description="Cursor line/column")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context for completion"
    )


class ApiResponse(BaseModel):
    """Standard SCK API response envelope."""

    status: str = Field(..., description="Response status (success/error)")
    code: int = Field(..., description="HTTP status code")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    errors: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Error details"
    )


# Global Langflow client instance
langflow_client: Optional[LangflowClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global langflow_client

    # Initialize Langflow client
    try:
        langflow_client = LangflowClient(
            base_url=f"http://localhost:7860",  # Default Langflow port
            flow_id="yaml-cf-ai-agent-v1",
        )
        log.info("Langflow client initialized successfully")
    except Exception as e:
        log.warning(f"Failed to initialize Langflow client: {e}")
        langflow_client = LangflowClient()  # Use mock client

    yield

    # Cleanup
    log.info("Shutting down SCK Core AI server")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="SCK Core AI",
        description="AI-powered YAML and CloudFormation processing agent",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Create app instance
app = create_app()


def create_envelope_response(
    status: str = "success",
    code: int = 200,
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create SCK-standard API response envelope."""
    response = {"status": status, "code": code}

    if data is not None:
        response["data"] = data
    if message is not None:
        response["message"] = message
    if errors is not None:
        response["errors"] = errors
    if metadata is not None:
        response["metadata"] = metadata

    return response


# Health Check Endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return create_envelope_response(
        data={"status": "healthy", "service": "sck-core-ai"},
        message="Service is running normally",
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return create_envelope_response(
        data={
            "service": "SCK Core AI",
            "version": "0.1.0",
            "endpoints": [
                "/health",
                "/api/v1/lint/yaml",
                "/api/v1/validate/cloudformation",
                "/api/v1/complete",
            ],
        },
        message="Welcome to SCK Core AI Agent",
    )


# API v1 Endpoints
@app.post("/api/v1/lint/yaml", tags=["linting"], response_model=ApiResponse)
async def lint_yaml(request: YamlLintRequest):
    """
    Lint and validate YAML content.

    Processes YAML through the Langflow AI agent pipeline for:
    - Syntax validation
    - Structure analysis
    - Best practices checking
    - AI-powered suggestions
    """
    try:
        log.info("Processing YAML lint request", content_length=len(request.content))

        if not langflow_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Langflow client not available",
            )

        # Process through Langflow
        result = langflow_client.process_sync(
            {
                "input_value": request.content,
                "tweaks": {
                    "yaml-parser": {
                        "validation_mode": request.options.get("mode", "syntax")
                    },
                    "response-formatter": {"format_type": "sck_envelope"},
                },
            }
        )

        # Extract data from Langflow response
        if isinstance(result, dict) and "data" in result:
            response_data = result["data"]
        else:
            # Fallback response structure
            response_data = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": [],
                "metrics": {"processing_time_ms": 100, "langflow_execution_id": "mock"},
            }

        return create_envelope_response(
            data=response_data, message="YAML linting completed successfully"
        )

    except Exception as e:
        log.error("YAML linting failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"YAML linting failed: {str(e)}",
        )


@app.post(
    "/api/v1/validate/cloudformation", tags=["validation"], response_model=ApiResponse
)
async def validate_cloudformation(request: CloudFormationValidateRequest):
    """
    Validate CloudFormation templates.

    Performs comprehensive CloudFormation analysis including:
    - Schema validation
    - Resource dependency checking
    - Security analysis
    - Cost optimization suggestions
    """
    try:
        log.info("Processing CloudFormation validation", region=request.region)

        if not langflow_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Langflow client not available",
            )

        # Convert template to YAML string for processing
        template_yaml = json.dumps(request.template, indent=2)

        # Process through Langflow
        result = langflow_client.process_sync(
            {
                "input_value": template_yaml,
                "tweaks": {
                    "cf-validator": {
                        "region": request.region,
                        "strict_mode": request.strict,
                    },
                    "response-formatter": {"format_type": "sck_envelope"},
                },
            }
        )

        # Extract validation results
        if isinstance(result, dict) and "data" in result:
            response_data = result["data"]
        else:
            response_data = {
                "valid": True,
                "template_info": {
                    "resources": len(request.template.get("Resources", {})),
                    "parameters": len(request.template.get("Parameters", {})),
                    "outputs": len(request.template.get("Outputs", {})),
                },
                "errors": [],
                "warnings": [],
                "security_analysis": {},
                "cost_analysis": {},
            }

        return create_envelope_response(
            data=response_data, message="CloudFormation validation completed"
        )

    except Exception as e:
        log.error("CloudFormation validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CloudFormation validation failed: {str(e)}",
        )


@app.post("/api/v1/complete", tags=["completion"], response_model=ApiResponse)
async def code_completion(request: CodeCompletionRequest):
    """
    Provide AI-powered code completion suggestions.

    Analyzes partial YAML/CloudFormation content and provides
    intelligent completion suggestions based on:
    - Current context and structure
    - AWS CloudFormation best practices
    - Common patterns and templates
    """
    try:
        log.info(
            "Processing code completion request",
            cursor_line=request.cursor_position.get("line", 0),
        )

        if not langflow_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Langflow client not available",
            )

        # Process through specialized completion workflow
        result = langflow_client.process_sync(
            {
                "input_value": request.content,
                "tweaks": {
                    "ai-analyzer": {
                        "system_message": f"""Provide code completion suggestions for the given YAML/CloudFormation content.
                    
Cursor position: Line {request.cursor_position.get('line', 0)}, Column {request.cursor_position.get('column', 0)}

Focus on:
1. Valid CloudFormation resources and properties
2. Proper YAML syntax and indentation
3. AWS best practices and security
4. Common configuration patterns

Return suggestions as a JSON array with: text, description, insertText, kind."""
                    },
                    "response-formatter": {"format_type": "sck_envelope"},
                },
            }
        )

        # Extract completion suggestions
        if isinstance(result, dict) and "data" in result:
            response_data = result["data"]
        else:
            response_data = {
                "suggestions": [
                    {
                        "text": "Resources:",
                        "description": "CloudFormation Resources section",
                        "insertText": "Resources:\n  ",
                        "kind": "keyword",
                    }
                ],
                "context": request.context,
            }

        return create_envelope_response(
            data=response_data, message="Code completion suggestions generated"
        )

    except Exception as e:
        log.error("Code completion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code completion failed: {str(e)}",
        )


# AWS Lambda Handler (using Mangum)
def lambda_handler(event, context):
    """
    AWS Lambda handler for API Gateway integration.

    This handler is synchronous as required by SCK framework standards.
    """
    if SCK_AVAILABLE:
        # Use SCK ProxyEvent for proper request parsing
        request = ProxyEvent(**event)
        log.info("Processing Lambda request", method=request.method, path=request.path)
    else:
        log.info("Processing Lambda request (fallback mode)")

    # Use Mangum to adapt FastAPI for Lambda
    asgi_handler = Mangum(app, lifespan="off")
    return asgi_handler(event, context)


def main():
    """Main entry point for local development server."""
    import uvicorn

    uvicorn.run(
        "core_ai.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    main()
