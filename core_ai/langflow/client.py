"""
Langflow client for SCK Core AI agent.

Provides synchronous interface to Langflow workflows for Lambda compatibility.
"""

import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import core_logging as log

    SCK_AVAILABLE = True
except ImportError:
    import logging as log

    SCK_AVAILABLE = False

# Configure logging
if SCK_AVAILABLE:
    logger = log.get_logger(__name__)
else:
    logger = log.getLogger(__name__)


class LangflowClient:
    """
    Synchronous Langflow client for Lambda compatibility.

    Provides interface to Langflow workflows while maintaining
    SCK framework patterns (no async in Lambda handlers).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        flow_id: str = "yaml-cf-ai-agent-v1",
        timeout: int = 30,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Langflow client.

        Args:
            base_url: Langflow server URL
            flow_id: Default flow ID to execute
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.flow_id = flow_id
        self.timeout = timeout
        self.api_key = api_key

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, using mock responses")

        # Build API endpoints
        self.api_base = f"{self.base_url}/api/v1"
        self.flows_endpoint = f"{self.api_base}/flows"

        logger.info(
            "Langflow client initialized", base_url=self.base_url, flow_id=self.flow_id
        )

    def process_sync(
        self,
        inputs: Dict[str, Any],
        flow_id: Optional[str] = None,
        tweaks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process inputs through Langflow workflow synchronously.

        Args:
            inputs: Input data for the workflow
            flow_id: Override default flow ID
            tweaks: Runtime configuration overrides

        Returns:
            Workflow execution results

        Raises:
            LangflowError: If workflow execution fails
        """
        target_flow_id = flow_id or self.flow_id
        start_time = time.time()

        logger.info(
            "Starting Langflow processing",
            flow_id=target_flow_id,
            inputs_keys=list(inputs.keys()),
        )

        if not HTTPX_AVAILABLE:
            return self._mock_response(inputs, target_flow_id)

        try:
            # Prepare request payload
            payload = {
                "input_value": inputs.get("input_value", ""),
                "output_type": "chat",
                "input_type": "chat",
                "tweaks": tweaks or {},
            }

            # Set up headers
            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Execute workflow
            url = f"{self.flows_endpoint}/{target_flow_id}"

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()

                processing_time = time.time() - start_time
                logger.info(
                    "Langflow processing completed",
                    flow_id=target_flow_id,
                    processing_time_ms=int(processing_time * 1000),
                )

                return self._format_response(result, processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Langflow processing failed",
                flow_id=target_flow_id,
                error=str(e),
                processing_time_ms=int(processing_time * 1000),
            )

            return {
                "status": "error",
                "code": 500,
                "message": f"Langflow processing failed: {str(e)}",
                "data": None,
                "metadata": {
                    "flow_id": target_flow_id,
                    "processing_time_ms": int(processing_time * 1000),
                    "error_type": type(e).__name__,
                },
            }

    def _format_response(
        self, raw_result: Any, processing_time: float
    ) -> Dict[str, Any]:
        """
        Format Langflow response into SCK envelope structure.

        Args:
            raw_result: Raw response from Langflow
            processing_time: Processing duration in seconds

        Returns:
            Formatted SCK response envelope
        """
        try:
            # Extract meaningful data from Langflow response
            # Note: Langflow response structure may vary, adapt as needed
            if isinstance(raw_result, dict):
                outputs = raw_result.get("outputs", [])
                if outputs and len(outputs) > 0:
                    # Get first output result
                    first_output = outputs[0]
                    output_data = first_output.get("outputs", [{}])[0]
                    result_text = (
                        output_data.get("results", {})
                        .get("message", {})
                        .get("text", "")
                    )

                    # Try to parse as JSON if it looks like structured data
                    try:
                        parsed_data = json.loads(result_text)
                        if isinstance(parsed_data, dict):
                            return parsed_data
                    except (json.JSONDecodeError, TypeError):
                        pass

                    # Return as text data
                    return {
                        "status": "success",
                        "code": 200,
                        "data": {"result": result_text, "raw_output": first_output},
                        "metadata": {
                            "flow_id": self.flow_id,
                            "processing_time_ms": int(processing_time * 1000),
                            "langflow_session": raw_result.get("session_id"),
                        },
                    }

            # Fallback for unexpected response format
            return {
                "status": "success",
                "code": 200,
                "data": {"raw_result": raw_result},
                "metadata": {
                    "flow_id": self.flow_id,
                    "processing_time_ms": int(processing_time * 1000),
                },
            }

        except Exception as e:
            logger.error("Response formatting failed", error=str(e))
            return {
                "status": "error",
                "code": 500,
                "message": f"Response formatting failed: {str(e)}",
                "data": {"raw_result": raw_result},
                "metadata": {
                    "flow_id": self.flow_id,
                    "processing_time_ms": int(processing_time * 1000),
                },
            }

    def _mock_response(self, inputs: Dict[str, Any], flow_id: str) -> Dict[str, Any]:
        """
        Generate mock response for development/testing.

        Args:
            inputs: Input data
            flow_id: Flow identifier

        Returns:
            Mock response in SCK envelope format
        """
        content = inputs.get("input_value", "")

        # Generate appropriate mock based on content
        mock_data = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [
                {
                    "line": 1,
                    "message": "Consider adding description field",
                    "type": "suggestion",
                }
            ],
            "analysis": {
                "content_type": "yaml",
                "lines": len(content.split("\n")) if content else 0,
                "complexity": "low",
            },
        }

        # Add CloudFormation-specific mock data if it looks like CF
        if any(
            keyword in content.lower()
            for keyword in ["resources:", "aws::", "cloudformation"]
        ):
            mock_data["cloudformation"] = {
                "template_version": "2010-09-09",
                "resource_count": 2,
                "security_score": 85,
                "cost_estimate": "low",
            }

        return {
            "status": "success",
            "code": 200,
            "data": mock_data,
            "metadata": {
                "flow_id": flow_id,
                "processing_time_ms": 150,
                "mock_response": True,
            },
            "message": "Mock response (Langflow not available)",
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check Langflow server health.

        Returns:
            Health status information
        """
        if not HTTPX_AVAILABLE:
            return {
                "status": "mock",
                "available": False,
                "message": "httpx not available",
            }

        try:
            health_url = f"{self.base_url}/health"

            with httpx.Client(timeout=5) as client:
                response = client.get(health_url)
                response.raise_for_status()

                return {
                    "status": "healthy",
                    "available": True,
                    "langflow_version": response.headers.get(
                        "X-Langflow-Version", "unknown"
                    ),
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                }

        except Exception as e:
            return {"status": "unhealthy", "available": False, "error": str(e)}

    def list_flows(self) -> Dict[str, Any]:
        """
        List available flows on Langflow server.

        Returns:
            Available flows information
        """
        if not HTTPX_AVAILABLE:
            return {"flows": [{"id": self.flow_id, "name": "Mock Flow"}], "mock": True}

        try:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(timeout=10) as client:
                response = client.get(self.flows_endpoint, headers=headers)
                response.raise_for_status()

                flows_data = response.json()
                return {
                    "flows": flows_data,
                    "count": len(flows_data) if isinstance(flows_data, list) else 1,
                }

        except Exception as e:
            logger.error("Failed to list flows", error=str(e))
            return {"flows": [], "error": str(e)}


class LangflowError(Exception):
    """Exception raised for Langflow-related errors."""

    pass
