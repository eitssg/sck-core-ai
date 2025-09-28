"""Test configuration and fixtures for SCK Core AI."""

import os
import pytest
from unittest.mock import Mock, patch

# Set test environment
os.environ["LOCAL_MODE"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture
def client():
    """FastAPI test client with Langflow patched to deterministic stub.

    We patch both the original client class location and the re-export inside
    `core_ai.server` (which imports the class name directly) because the server
    module may have been imported earlier during test collection, binding the
    original class object. We also proactively assign the global
    `server.langflow_client` after importing the app so the lifespan initializer
    doesn't attempt a real network call to Langflow.
    """
    # Build stub object
    stub = Mock()
    stub.process_sync.return_value = {
        "status": "success",
        "code": 200,
        "data": {"text": "TEST_REPLY", "result": "TEST_REPLY"},
        "metadata": {"mock": True},
    }
    stub.health_check.return_value = {"status": "healthy", "available": True}

    # Define a trivial constructor that returns the stub (used for class patch)
    class _StubLangflow:
        def __init__(self, *a, **k):  # pragma: no cover - simple container
            pass

        def process_sync(self, *a, **k):  # delegate
            return stub.process_sync.return_value

        def health_check(self):  # delegate
            return stub.health_check.return_value

    with (
        patch("core_ai.langflow.client.LangflowClient", _StubLangflow),
        patch("core_ai.server.LangflowClient", _StubLangflow),
    ):
        from fastapi.testclient import TestClient
        from core_ai import server
        from core_ai.server import app

        # Ensure global reference used by request handlers points at stub
        server.langflow_client = _StubLangflow()

        with TestClient(app) as test_client:
            yield test_client


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
