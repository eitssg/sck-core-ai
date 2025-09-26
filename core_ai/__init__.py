"""
SCK Core AI - Intelligent YAML/CloudFormation Agent

An AI-powered service for YAML and CloudFormation template linting,
validation, and code completion using Langflow workflows.
"""

__version__ = "0.1.0"
__author__ = "James Barwick"
__email__ = "jbarwick@me.com"

from .server import create_app
from .mcp_server import create_mcp_server

__all__ = ["create_app", "create_mcp_server"]
