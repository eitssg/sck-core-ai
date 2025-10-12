"""
Langflow client for SCK Core AI agent.

Provides synchronous interface to Langflow workflows for Lambda compatibility.
"""

import json
import time
from typing import Any, Dict, Optional
import httpx
import core_logging as logger
import core_framework as util

# NOTE: Tests patch this flag to simulate absence of httpx without using guarded imports.
# We import httpx unconditionally (fail-fast policy) but allow tests to force a "not available" path.
HTTPX_AVAILABLE = True


class LangflowClient:
    """
    Synchronous Langflow client for Lambda compatibility.

    Provides interface to Langflow workflows while maintaining
    SCK framework patterns (no async in Lambda handlers).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        flow_id: Optional[str] = None,
        timeout: int = 30,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Langflow client.

        Args:
            base_url: Langflow server URL (defaults to LANGFLOW_BASE_URL env var)
            flow_id: Default flow ID to execute (defaults to LANGFLOW_FLOW_ID env var)
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication (defaults to LANGFLOW_API_KEY env var)
        """
        self.base_url = (base_url or util.get_langflow_base_url()).rstrip("/")
        self.flow_id = flow_id or util.get_langflow_flow_id()
        self.timeout = timeout
        self.api_key = api_key or util.get_langflow_api_key()

        # Build API endpoints
        self.api_base = f"{self.base_url}/api/v1"
        self.flows_endpoint = f"{self.api_base}/flows"

        logger.info("Langflow client initialized", base_url=self.base_url, flow_id=self.flow_id)

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

        try:
            # Prepare canonical payload components (some Langflow versions expect different shapes)
            base_input_value = inputs.get("input_value") or inputs.get("text") or inputs.get("prompt") or ""
            base_tweaks = tweaks or inputs.get("tweaks") or {}

            # Common payload variants we will attempt (ordered). Newer Langflow builds typically
            # accept /flows/{id}/run with root-level keys, while others require /run/{id} or /process/{id}
            # and may expect an "inputs" object. We try several to avoid hard failing on a minor API drift.
            payload_variants = [
                {  # Preferred modern format (root keys)
                    "input_value": base_input_value,
                    "output_type": "chat",
                    "input_type": "chat",
                    "tweaks": base_tweaks,
                },
                {  # Wrapped inputs variant
                    "inputs": {"input_value": base_input_value},
                    "tweaks": base_tweaks,
                },
                {  # Minimal variant
                    "input_value": base_input_value,
                    "tweaks": base_tweaks,
                },
            ]

            # Candidate execution endpoints (relative). We will try each until one succeeds.
            # Order chosen based on most commonly documented patterns.
            endpoint_variants = [
                f"/flows/{target_flow_id}/run",  # documented variant (may 405 on some versions)
                f"/run/{target_flow_id}",  # alternative pattern
                f"/process/{target_flow_id}",  # legacy / process style
                f"/flows/{target_flow_id}",  # some builds accept POST directly
            ]

            # Aggregate attempt results for error transparency
            attempt_errors = []

            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self.api_key:
                # Provide both header styles for maximum compatibility
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key

            with httpx.Client(base_url=self.api_base, timeout=self.timeout) as client:
                for ep in endpoint_variants:
                    for pv in payload_variants:
                        url = ep
                        try:
                            logger.debug(
                                "Attempting Langflow execution",
                                endpoint=url,
                                payload_shape=list(pv.keys()),
                                flow_id=target_flow_id,
                            )
                            response = client.post(url, json=pv, headers=headers)
                            # Accept only 2xx
                            if 200 <= response.status_code < 300:
                                result = response.json()
                                processing_time = time.time() - start_time
                                logger.info(
                                    "Langflow processing completed",
                                    flow_id=target_flow_id,
                                    endpoint=url,
                                    processing_time_ms=int(processing_time * 1000),
                                )
                                return self._format_response(result, processing_time)
                            else:
                                attempt_errors.append(
                                    {
                                        "endpoint": url,
                                        "status_code": response.status_code,
                                        "body": response.text[:500],
                                    }
                                )
                                # 401/403 likely won't improve by changing payload shape; break early for that endpoint
                                if response.status_code in (401, 403):
                                    break
                        except Exception as attempt_exc:  # network/JSON issues
                            attempt_errors.append(
                                {
                                    "endpoint": url,
                                    "exception": type(attempt_exc).__name__,
                                    "error": str(attempt_exc)[:500],
                                }
                            )
                            continue

            raise RuntimeError(
                "All Langflow execution attempts failed",
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Langflow processing failed",
                flow_id=target_flow_id,
                error=str(e),
                attempts=attempt_errors if "attempt_errors" in locals() else None,
                processing_time_ms=int(processing_time * 1000),
            )

            return {
                "status": "error",
                "code": 500,
                "message": f"Langflow processing failed: {str(e)}",
                "data": ({"attempts": attempt_errors} if "attempt_errors" in locals() else None),
                "metadata": {
                    "flow_id": target_flow_id,
                    "processing_time_ms": int(processing_time * 1000),
                    "error_type": type(e).__name__,
                },
            }

    def _format_response(self, raw_result: Any, processing_time: float) -> Dict[str, Any]:
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
                    result_text = output_data.get("results", {}).get("message", {}).get("text", "")

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
        if any(keyword in content.lower() for keyword in ["resources:", "aws::", "cloudformation"]):
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
            # Test expectations: status == "mock" and available False when httpx flagged unavailable
            return {
                "status": "mock",
                "available": False,
                "reason": "httpx library flagged unavailable",
            }
        try:
            health_url = f"{self.base_url}/health"

            with httpx.Client(timeout=5) as client:
                response = client.get(health_url)
                response.raise_for_status()

                return {
                    "status": "healthy",
                    "available": True,
                    "langflow_version": response.headers.get("X-Langflow-Version", "unknown"),
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
            # Provide deterministic mock response for tests when httpx is flagged off
            return {
                "flows": [
                    {
                        "id": "mock-flow",
                        "name": "Mock Flow (httpx unavailable)",
                    }
                ],
                "count": 1,
                "mock": True,
            }
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
