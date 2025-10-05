from typing import Dict
import uuid

from datetime import datetime

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

clients: Dict[str, OAuthClientInformationFull] = {}
authcode: Dict[str, AuthorizationCode] = {}

oauth_tokens: Dict[str, OAuthToken] = {}

accesstokens: Dict[str, AccessToken] = {}
refreshtokens: Dict[str, RefreshToken] = {}


class AuthProvider(OAuthAuthorizationServerProvider):

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        code = str(uuid.uuid4())
        auth_code = AuthorizationCode(
            code=code,
            client_id=client.client_id,
            redirect_uri=params.redirect_uri,
            scopes=params.scopes or [],
            code_challenge=params.code_challenge,
            redirect_uri_provided_explicitly=bool(params.redirect_uri),
            expires_at=int(int(datetime.now().timestamp()) + 600),  # 10 minutes from now
        )
        authcode[client.client_id] = auth_code
        return code

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return authcode.get(client.client_id)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:

        access_token = AccessToken(
            token=str(uuid.uuid4()),
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(int(datetime.now().timestamp()) + 3600),  # 1 hour from now
        )

        accesstokens[access_token.token] = access_token

        refresh_token = RefreshToken(
            token=str(uuid.uuid4()),
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(int(datetime.now().timestamp()) + 86400),  # 24 hours from now
        )

        refreshtokens[refresh_token.token] = refresh_token

        token = OAuthToken(
            access_token=access_token.token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token.token,
            scope=" ".join(authorization_code.scopes),
        )
        oauth_tokens[client.client_id] = token
        return token

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        """
        Loads a RefreshToken by its token string.

        Args:
            client: The client that is requesting to load the refresh token.
            refresh_token: The refresh token string to load.

        Returns:
            The RefreshToken object if found, or None if not found.
        """
        return refreshtokens.get(refresh_token)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """
        Exchanges a refresh token for an access token and refresh token.

        Implementations SHOULD rotate both the access token and refresh token.

        Args:
            client: The client exchanging the refresh token.
            refresh_token: The refresh token to exchange.
            scopes: Optional scopes to request with the new access token.

        Returns:
            The OAuth token, containing access and refresh tokens.

        Raises:
            TokenError: If the request is invalid
        """
        st: RefreshToken | None = refreshtokens.get(refresh_token.token)

        if st is None:
            raise Exception("Invalid refresh token")  # Replace with appropriate exception

        access_token = AccessToken(
            token=str(uuid.uuid4()),
            client_id=client.client_id,
            scopes=scopes,
            expires_at=int(int(datetime.now().timestamp()) + 3600),  # 1 hour from now
        )

        accesstokens[access_token.token] = access_token

        refresh_token = RefreshToken(
            token=str(uuid.uuid4()),
            client_id=client.client_id,
            scopes=scopes,
            expires_at=int(int(datetime.now().timestamp()) + 86400),  # 24 hours from now
        )

        refreshtokens[refresh_token.token] = refresh_token

        token = OAuthToken(
            access_token=access_token.token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token.token,
            scope=" ".join(scopes),
        )
        oauth_tokens[client.client_id] = token
        return token

    async def load_access_token(self, token: str) -> AccessToken | None:
        """
        Loads an access token by its token.

        Args:
            token: The access token to verify.

        Returns:
            The AuthInfo, or None if the token is invalid.
        """
        return accesstokens.get(token)

    async def revoke_token(
        self,
        token: AccessToken | RefreshToken,
    ) -> None:
        """
        Revokes an access or refresh token.

        If the given token is invalid or already revoked, this method should do nothing.

        Implementations SHOULD revoke both the access token and its corresponding
        refresh token, regardless of which of the access token or refresh token is
        provided.

        Args:
            token: the token to revoke
        """
        if isinstance(token, AccessToken):
            if token.token in accesstokens:
                del accesstokens[token.token]
        elif isinstance(token, RefreshToken):
            if token.token in refreshtokens:
                del refreshtokens[token.token]
