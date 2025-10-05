"""
Simple SCK Tools for Langflow
Direct integration with SCK components without MCP bridge complexity
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class SCKArchitectureTool:
    """Tool to provide SCK architecture information."""

    @staticmethod
    def get_architecture_info(component: str = "") -> str:
        """Get SCK architecture information."""

        architecture_info = {
            "overview": """
SCK (Simple Cloud Kit) Architecture Overview:

🏗️ **Core Execution Chain:**
CLI/UI → core-invoker → core-runner → [core-deployspec, core-component]

📦 **Python Modules (17+ submodules):**
- sck-core-framework: Base utilities, configuration, data models
- sck-core-logging: Structured logging with correlation IDs  
- sck-core-db: Database interface, DynamoDB/PynamoDB helpers
- sck-core-execute: Lambda step-function execution engine
- sck-core-runner: Kicks off core_execute step functions
- sck-core-deployspec: Generates and compiles deployment specs
- sck-core-component: Manages components in S3 and DynamoDB
- sck-core-invoker: Orchestrates lambda function execution
- sck-core-organization: AWS Organizations, SCPs, Accounts
- sck-core-api: FastAPI dev server, AWS API Gateway prod
- sck-core-codecommit: AWS CodeCommit event handling
- sck-core-cli: Command-line interface

🗄️ **S3 Architecture:**
Three bucket prefixes with lifecycle management:
- packages/: Input files (Jinja2 templates, deployment packages)
- files/: Pre-compilation resources and post-compilation files
- artefacts/: Post-compilation CloudFormation templates

⚡ **Lambda Runtime Model:**
- All Python runs in AWS Lambda (synchronous handlers only)
- Use ProxyEvent(**event) for API Gateway integration
- NO async/await in Lambda code
- Use MagicS3Bucket for bucket operations, boto3 client for presigned URLs
            """,
            "imports": """
📋 **Standard SCK Import Patterns:**

```python
# Core imports (MANDATORY)
import core_framework as util
import core_logging as log
import core_helper.aws as aws
from core_framework import ProxyEvent

# Logging setup (CRITICAL)
logger = log  # NOT log.get_logger()

# S3 operations
from core_helper.magic import MagicS3Bucket
bucket = MagicS3Bucket(bucket_name=util.get_bucket_name(), region=util.get_bucket_region())
```
            """,
            "lambda": """
🚀 **Lambda Handler Pattern:**

```python
from core_framework import ProxyEvent
import core_logging as log
import json

logger = log

def lambda_handler(event, context):
    \"\"\"Standard SCK Lambda handler.\"\"\"
    request = ProxyEvent(**event)
    logger.info("Processing request", extra={"path": request.path})
    
    # Process request
    result = {"message": "Success"}
    
    # Return envelope response
    return {
        "statusCode": 200,
        "body": json.dumps({
            "status": "success",
            "code": 200,
            "data": result,
            "metadata": {},
            "message": "Request processed successfully"
        })
    }
```
            """,
            "s3": """
🗄️ **S3 Operations Pattern:**

```python
from core_helper.magic import MagicS3Bucket
import core_framework as util
import boto3

# For bucket operations (put, get, list)
bucket = MagicS3Bucket(
    bucket_name=util.get_bucket_name(), 
    region=util.get_bucket_region()
)
bucket.put_object(Key="packages/template.yaml", Body=content)

# For presigned URLs (MagicS3Bucket doesn't support this)
s3_client = boto3.client('s3', region_name=util.get_bucket_region())
presigned_url = s3_client.generate_presigned_url(
    'put_object',
    Params={'Bucket': util.get_bucket_name(), 'Key': 'files/upload.zip'},
    ExpiresIn=3600
)
```
            """,
        }

        if component.lower() in architecture_info:
            return architecture_info[component.lower()]
        elif component:
            return f"Component '{component}' not found. Available: {', '.join(architecture_info.keys())}"
        else:
            return architecture_info["overview"]
