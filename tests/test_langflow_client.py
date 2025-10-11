"""Test cases for Langflow client integration."""

import json
from unittest.mock import Mock, patch

from core_ai.langflow.client import LangflowClient


class TestLangflowClient:
    """Test Langflow client functionality."""

    def test_init_default_params(self):
        """Test client initialization with default parameters."""
        client = LangflowClient()

        assert client.base_url == "http://localhost:7860"
        assert client.flow_id == "yaml-cf-ai-agent-v1"
        assert client.timeout == 30
        assert client.api_key is None

    def test_init_custom_params(self):
        """Test client initialization with custom parameters."""
        client = LangflowClient(
            base_url="https://langflow.example.com",
            flow_id="custom-flow-id",
            timeout=60,
            api_key="test-key",
        )

        assert client.base_url == "https://langflow.example.com"
        assert client.flow_id == "custom-flow-id"
        assert client.timeout == 60
        assert client.api_key == "test-key"

    def test_mock_response_yaml(self):
        """Test mock response generation for YAML content."""
        client = LangflowClient()

        inputs = {"input_value": "key: value\nlist:\n  - item1\n  - item2"}
        result = client._mock_response(inputs, "test-flow")

        assert result["status"] == "success"
        assert result["code"] == 200
        assert "data" in result
        assert result["data"]["valid"] is True
        assert "suggestions" in result["data"]
        assert result["metadata"]["mock_response"] is True

    def test_mock_response_cloudformation(self):
        """Test mock response for CloudFormation content."""
        client = LangflowClient()

        cf_content = """
        AWSTemplateFormatVersion: '2010-09-09'
        Resources:
          MyBucket:
            Type: AWS::S3::Bucket
        """

        inputs = {"input_value": cf_content}
        result = client._mock_response(inputs, "test-flow")

        assert result["status"] == "success"
        assert "cloudformation" in result["data"]
        assert result["data"]["cloudformation"]["resource_count"] == 2
        assert "security_score" in result["data"]["cloudformation"]

    @patch("core_ai.langflow.client.httpx")
    def test_process_sync_success(self, mock_httpx):
        """Test successful workflow processing."""
        # Mock httpx response
        mock_response = Mock()
        mock_response.json.return_value = {
            "outputs": [{"outputs": [{"results": {"message": {"text": json.dumps({"valid": True, "errors": []})}}}]}]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = LangflowClient()
        inputs = {"input_value": "test: yaml"}

        result = client.process_sync(inputs)

        assert "valid" in result, f"Check for errors {result['message']}"
        assert result["valid"] is True
        assert result["errors"] == []

    @patch("core_ai.langflow.client.httpx")
    def test_process_sync_http_error(self, mock_httpx):
        """Test workflow processing with HTTP error."""
        # Mock httpx to raise exception
        mock_client = Mock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = LangflowClient()
        inputs = {"input_value": "test: yaml"}

        result = client.process_sync(inputs)

        assert result["status"] == "error"
        assert result["code"] == 500
        assert "Connection error" in result["message"]

    def test_health_check_no_httpx(self):
        """Test health check when httpx is not available."""
        with patch("core_ai.langflow.client.HTTPX_AVAILABLE", False):
            client = LangflowClient()
            result = client.health_check()

            assert result["status"] == "mock"
            assert result["available"] is False

    @patch("core_ai.langflow.client.httpx")
    def test_health_check_success(self, mock_httpx):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"X-Langflow-Version": "1.0.0"}
        mock_response.elapsed.total_seconds.return_value = 0.1

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = LangflowClient()
        result = client.health_check()

        assert result["status"] == "healthy"
        assert result["available"] is True
        assert result["langflow_version"] == "1.0.0"

    def test_list_flows_no_httpx(self):
        """Test list flows when httpx is not available."""
        with patch("core_ai.langflow.client.HTTPX_AVAILABLE", False):
            client = LangflowClient()
            result = client.list_flows()

            assert result["mock"] is True
            assert len(result["flows"]) == 1

    @patch("core_ai.langflow.client.httpx")
    def test_list_flows_success(self, mock_httpx):
        """Test successful flows listing."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "flow1", "name": "Flow 1"},
            {"id": "flow2", "name": "Flow 2"},
        ]
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        client = LangflowClient()
        result = client.list_flows()

        assert result["count"] == 2
        assert len(result["flows"]) == 2
        assert result["flows"][0]["id"] == "flow1"

    def test_format_response_valid_json(self):
        """Test response formatting with valid JSON text."""
        client = LangflowClient()

        raw_result = {"outputs": [{"outputs": [{"results": {"message": {"text": '{"valid": true, "errors": []}'}}}]}]}

        result = client._format_response(raw_result, 0.5)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_format_response_invalid_json(self):
        """Test response formatting with invalid JSON text."""
        client = LangflowClient()

        raw_result = {"outputs": [{"outputs": [{"results": {"message": {"text": "This is plain text, not JSON"}}}]}]}

        result = client._format_response(raw_result, 0.5)

        assert result["status"] == "success"
        assert result["data"]["result"] == "This is plain text, not JSON"

    def test_format_response_unexpected_format(self):
        """Test response formatting with unexpected structure."""
        client = LangflowClient()

        raw_result = "unexpected string response"

        result = client._format_response(raw_result, 0.5)

        assert result["status"] == "success"
        assert result["data"]["raw_result"] == "unexpected string response"
