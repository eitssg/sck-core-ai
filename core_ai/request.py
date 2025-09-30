from typing import Any, Dict, List, Optional, Union
import json
import urllib.parse
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo

import core_framework as util

DOMAIN_PREFIX = (
    "your-api-gateway-domain-prefix"  # Replace with your actual domain prefix
)
API_ID = "your-api-id"  # Replace with your actual API ID


class RequestContext(BaseModel):
    """AWS API Gateway request context information.

    Contains comprehensive metadata about the API Gateway request, including
    routing information, timing data, identity context, and AWS-specific
    identifiers. This matches the requestContext structure that AWS API Gateway
    provides to Lambda functions.

    Attributes:
        resourceId (str): API Gateway resource identifier.
        resourcePath (str): Resource path template with parameter placeholders.
        httpMethod (str): HTTP method (GET, POST, PUT, DELETE, etc.).
        extendedRequestId (Optional[str]): Extended request ID for detailed tracing.
        requestTime (str): Human-readable request timestamp.
        path (str): Full request path including stage prefix.
        accountId (Optional[str]): AWS account ID for the API Gateway.
        protocol (str): HTTP protocol version (default: "HTTP/1.1").
        stage (str): API Gateway deployment stage name.
        domainPrefix (str): Domain prefix for the API Gateway endpoint.
        requestTimeEpoch (int): Request timestamp as Unix epoch milliseconds.
        requestId (str): Unique identifier for this specific request.
        domainName (str): Full domain name of the API Gateway endpoint.
        identity (CognitoIdentity): Authentication and identity information.
        apiId (str): API Gateway API identifier.

    Note:
        The RequestContext provides complete metadata for:

        - Request routing and resource identification
        - Timing and tracing information
        - Authentication and authorization context
        - AWS infrastructure identifiers

    Example:
        .. code-block:: python

            context = RequestContext(
                resourceId="abc123",
                resourcePath="/users/{id}",
                httpMethod="GET",
                path="/prod/users/123",
                requestId="550e8400-e29b-41d4-a716-446655440000",
                identity=cognito_identity
            )

            # Access routing information
            print(f"Resource: {context.resourcePath}")
            print(f"Method: {context.httpMethod}")
            print(f"Stage: {context.stage}")
    """

    model_config = ConfigDict(populate_by_name=True)

    resourceId: str = Field(description="API Gateway resource identifier for routing")
    resourcePath: str = Field(
        description="Resource path template with parameter placeholders (e.g., '/users/{id}')"
    )
    httpMethod: str = Field(
        description="HTTP method for the request (GET, POST, PUT, DELETE, etc.)"
    )
    extendedRequestId: Optional[str] = Field(
        None, description="Extended request ID for detailed request tracing"
    )
    requestTime: str = Field(
        description="Human-readable request timestamp in API Gateway format",
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%d/%b/%Y:%H:%M:%S %z"
        ),
    )
    path: str = Field(
        description="Full request path including API Gateway stage prefix"
    )
    accountId: Optional[str] = Field(
        None, description="AWS account ID that owns the API Gateway"
    )
    protocol: str = Field(description="HTTP protocol version", default="HTTP/1.1")
    stage: str = Field(
        description="API Gateway deployment stage name (prod, dev, etc.)",
        default_factory=util.get_environment,
    )
    domainPrefix: str = Field(
        description="Domain prefix for the API Gateway endpoint", default=DOMAIN_PREFIX
    )
    requestTimeEpoch: int = Field(
        description="Request timestamp as Unix epoch time in milliseconds",
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000),
    )
    requestId: str = Field(
        description="Unique identifier for this specific API request"
    )
    domainName: str = Field(
        description="Full domain name of the API Gateway endpoint",
        default=f"{DOMAIN_PREFIX}.execute-api.us-east-1.amazonaws.com",
    )
    identity: dict = Field(
        description="Authentication and identity information for the request"
    )
    apiId: str = Field(description="API Gateway API identifier", default=API_ID)


class ProxyEvent(BaseModel):
    """AWS API Gateway proxy integration event model.

    Represents the complete event structure that AWS API Gateway sends to
    Lambda functions via proxy integration. This model ensures type safety
    and validation for all AWS API Gateway event fields.

    The body field is automatically parsed from JSON string to dictionary
    for convenient access in handler functions, while maintaining compatibility
    with the AWS event format.

    Attributes:
        httpMethod (str): HTTP method (GET, POST, PUT, DELETE, etc.).
        resource (str): API resource path with parameter placeholders.
        path (Optional[str]): Actual request path with resolved parameters.
        queryStringParameters (Dict[str, str]): Single-value query parameters.
        multiValueQueryStringParameters (Dict[str, List[str]]): Multi-value query parameters.
        pathParameters (Dict[str, str]): Path parameter values extracted from URL.
        stageVariables (Dict[str, str]): API Gateway stage variables.
        requestContext (RequestContext): Complete request context information.
        headers (Dict[str, str]): Single-value HTTP headers.
        multiValueHeaders (Dict[str, List[str]]): Multi-value HTTP headers.
        isBase64Encoded (bool): Whether the body content is base64 encoded.
        body (Union[Dict[str, Any], str]): Request body (auto-parsed from JSON).

    Note:
        AWS API Gateway always provides both single-value and multi-value
        versions of headers and query parameters. The multi-value versions
        are lists that can contain multiple values for the same key.

        The body field accepts both string (raw AWS format) and dict (parsed)
        formats, automatically converting JSON strings to dictionaries.

    Example:
        .. code-block:: python

            # From AWS API Gateway
            event = ProxyEvent(
                httpMethod="POST",
                resource="/users",
                path="/users",
                headers={"Content-Type": "application/json"},
                body='{"name": "John", "email": "john@example.com"}',
                requestContext=request_context
            )

            # Access parsed body
            user_data = event.body  # Returns: {"name": "John", "email": "john@example.com"}

            # Route key for handler lookup
            route = event.route_key  # Returns: "POST:/users"
    """

    model_config = ConfigDict(populate_by_name=True)

    httpMethod: str = Field(
        description="HTTP method for the request (GET, POST, PUT, DELETE, etc.)"
    )
    resource: str = Field(
        description="API resource path with parameter placeholders (e.g., '/users/{id}')"
    )
    path: Optional[str] = Field(
        None,
        description="Actual request path with resolved parameters (e.g., '/users/123')",
    )
    queryStringParameters: Dict[str, str] = Field(
        description="Single-value query string parameters", default_factory=dict
    )
    multiValueQueryStringParameters: Dict[str, List[str]] = Field(
        description="Multi-value query string parameters (AWS API Gateway format)",
        default_factory=dict,
    )
    pathParameters: Dict[str, str] = Field(
        description="Path parameter values extracted from the URL", default_factory=dict
    )
    stageVariables: Dict[str, str] = Field(
        description="API Gateway stage variables for environment configuration",
        default_factory=dict,
    )
    requestContext: RequestContext = Field(
        description="Complete request context information from API Gateway"
    )
    headers: Dict[str, str] = Field(
        description="Single-value HTTP request headers", default_factory=dict
    )
    multiValueHeaders: Dict[str, List[str]] = Field(
        description="Multi-value HTTP headers (AWS API Gateway format)",
        default_factory=dict,
    )
    cookies: Optional[list[str]] = Field(
        None, description="Parsed cookies from request (AWS API Gateway v2.0+ format)"
    )
    isBase64Encoded: bool = Field(
        description="Whether the body content is base64 encoded (for binary data)",
        default=False,
    )
    body: Union[Dict[str, Any], str] = Field(
        description="Request body content (automatically parsed from JSON string)",
        default_factory=dict,
    )

    @property
    def parsed_cookies(self) -> Dict[str, str]:
        """Parse cookies from headers if not provided in cookies field.

        Returns:
            Dict[str, str]: Parsed cookie name-value pairs

        Note:
            AWS API Gateway v1.0 puts cookies in headers['Cookie']
            AWS API Gateway v2.0+ puts them in the cookies field
        """
        # If API Gateway v2.0+ provided parsed cookies, use them
        if self.cookies:
            return {
                cookie.split("=", 1)[0]: cookie.split("=", 1)[1]
                for cookie in self.cookies
                if "=" in cookie
            }

        # Otherwise parse from Cookie header (v1.0 format)
        cookie_header = self.headers.get("Cookie", "")
        if not cookie_header:
            return {}

        cookies_list = cookie_header.split(";")
        return {
            cookie.split("=", 1)[0].strip(): cookie.split("=", 1)[1].strip()
            for cookie in cookies_list
            if "=" in cookie
        }

    @property
    def content_type(self) -> str:
        """Get the Content-Type header value."""
        return self.headers.get("content-type", "application/json")

    @field_validator("body", mode="after")
    @classmethod
    def body_dict(cls, body: Any, info: ValidationInfo) -> Union[Dict[str, Any], str]:
        """Convert JSON string body to dictionary for convenient access.

        Args:
            body (Any): Raw body value from API Gateway.
            info (ValidationInfo): Pydantic validation context.

        Returns:
            Union[Dict[str, Any], str]: Parsed dictionary or original string.

        Raises:
            ValueError: If JSON string is malformed.

        Note:
            - None values are converted to empty dictionaries
            - Valid dictionaries are passed through unchanged
            - JSON strings are parsed to dictionaries
            - Empty strings become empty dictionaries
            - Invalid JSON raises ValueError with descriptive message
        """
        if isinstance(body, dict):
            return body
        content_type = info.data.get("headers", {}).get(
            "content-type", "application/json"
        )
        if util.is_json_mimetype(content_type):
            try:
                return util.from_json(body) if body else {}
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string for body: {e}") from e
        elif "application/x-www-form-urlencoded" in content_type:
            data = cls._to_dict(urllib.parse.parse_qs(body)) if body else {}
            return data
        return {"data": body}  # Put whatever mimetype this data is in a dict

    @classmethod
    def _to_dict(cls, data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Convert form-encoded data to a dictionary.

        Args:
            data (Dict[str, List[str]]): Form-encoded data.

        Returns:
            Dict[str, Any]: Dictionary representation of the form data.
        """
        return {k: v[0] if len(v) == 1 else v for k, v in data.items()}

    @field_validator("httpMethod", mode="before")
    @classmethod
    def uppercase_method(cls, httpMethod: str, info: ValidationInfo) -> str:
        """Normalize HTTP method to uppercase for consistency.

        Args:
            httpMethod (str): HTTP method string.
            info (ValidationInfo): Pydantic validation context.

        Returns:
            str: Uppercase HTTP method.
        """
        return httpMethod.upper()

    @property
    def route_key(self) -> str:
        """Generate route key for handler lookup.

        Returns:
            str: Route key in format "METHOD:resource" for handler routing.

        Example:
            .. code-block:: python

                event = ProxyEvent(httpMethod="GET", resource="/users/{id}")
                route = event.route_key  # Returns: "GET:/users/{id}"
        """
        method = self.httpMethod.upper()
        return f"{method}:{self.resource}"

    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get header value case-insensitively.

        Args:
            name (str): Header name to retrieve.

        Returns:
            str: Header value or empty string if not found.
        """
        for k, v in self.headers.items():
            if k.lower() == name.lower():
                return k, v
        return name, default or None
