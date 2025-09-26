"""Test configuration and fixtures for SCK Core AI."""

import os
import pytest
from unittest.mock import Mock, patch

# Set test environment
os.environ["LOCAL_MODE"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture
def mock_langflow_client():
    """Mock Langflow client for testing."""
    with patch("core_ai.langflow.client.LangflowClient") as mock:
        client = Mock()
        client.process_sync.return_value = {
            "status": "success",
            "code": 200,
            "data": {"valid": True, "errors": [], "suggestions": []},
        }
        client.health_check.return_value = {"status": "healthy", "available": True}
        mock.return_value = client
        yield client


@pytest.fixture
def sample_yaml():
    """Sample YAML content for testing."""
    return """
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
data:
  key1: value1
  key2: value2
"""


@pytest.fixture
def sample_cloudformation():
    """Sample CloudFormation template for testing."""
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Test template",
        "Resources": {
            "TestBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {"BucketName": "test-bucket-name"},
            }
        },
    }


@pytest.fixture
def mock_fastapi_client():
    """Mock FastAPI test client."""
    from fastapi.testclient import TestClient
    from core_ai.server import app

    return TestClient(app)
