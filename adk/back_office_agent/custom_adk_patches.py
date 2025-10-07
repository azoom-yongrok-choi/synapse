"""
Custom ADK Patches for MCP Timeout Configuration.

This module provides custom implementations of ADK's MCP classes to allow
configurable timeouts for StdioServerParameters connections.

The google-adk 1.2.0 introduced a hardcoded 5-second timeout for stdio-based
MCP connections, which can be too short for some legitimate operations like
Spinach AI transcription and analysis.
"""

import sys
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, List, Optional, TextIO, Union

from google.adk.tools.mcp_tool.mcp_session_manager import (
    MCPSessionManager,
    StdioServerParameters,
    StreamableHTTPServerParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    ToolPredicate,
)
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

# Configure your desired timeout for stdio-based MCP connections
CUSTOM_STDIO_TIMEOUT_SECONDS = 300  # 60 seconds instead of the default 5 seconds


class CustomMcpSessionManager(MCPSessionManager):
    """
    Custom MCP Session Manager with configurable timeout for StdioServerParameters.

    This class overrides the create_session method to apply a custom timeout
    for stdio-based MCP connections, addressing the hardcoded 5-second limit
    introduced in google-adk 1.2.0.
    """

    def __init__(
        self,
        connection_params: Union[StdioServerParameters, StreamableHTTPServerParams],
        errlog: TextIO = sys.stderr,
    ):
        """Initialize the custom session manager with all required attributes."""
        # Initialize all attributes exactly as the original MCPSessionManager does
        self._connection_params = connection_params
        self._errlog = errlog
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    async def create_session(
        self,
        headers=None,
        timeout=None,
        sse_read_timeout=None,
        terminate_on_close=None,
        **kwargs,
    ) -> ClientSession:
        """
        Creates and initializes an MCP client session with custom timeout for StdioServerParameters.
        ADK 1.6.x 호환: headers, timeout 등 다양한 인자를 받을 수 있도록 확장.
        """
        if self._session is not None:
            return self._session

        # Create a new exit stack for this session
        self._exit_stack = AsyncExitStack()

        try:
            if isinstance(self._connection_params, StdioServerParameters):
                client = stdio_client(
                    server=self._connection_params, errlog=self._errlog
                )
            elif isinstance(self._connection_params, StreamableHTTPServerParams):
                client = streamablehttp_client(
                    url=self._connection_params.url,
                    headers=headers or self._connection_params.headers,
                    timeout=timeout
                    or timedelta(seconds=self._connection_params.timeout),
                    sse_read_timeout=sse_read_timeout
                    or timedelta(seconds=self._connection_params.sse_read_timeout),
                    terminate_on_close=terminate_on_close
                    if terminate_on_close is not None
                    else self._connection_params.terminate_on_close,
                )
            else:
                raise ValueError(
                    "Unable to initialize connection. Connection should be"
                    " StdioServerParameters, but got"
                    f" {self._connection_params}"
                )

            transports = await self._exit_stack.enter_async_context(client)

            # HERE IS THE CUSTOM TIMEOUT LOGIC:
            if isinstance(self._connection_params, StdioServerParameters):
                print(
                    f"CUSTOM_ADK: Applying custom timeout for StdioServerParameters: {CUSTOM_STDIO_TIMEOUT_SECONDS}s"
                )
                session = await self._exit_stack.enter_async_context(
                    ClientSession(
                        *transports[:2],
                        read_timeout_seconds=timedelta(
                            seconds=CUSTOM_STDIO_TIMEOUT_SECONDS
                        ),
                    )
                )
            else:
                # Original logic for other connection types
                session = await self._exit_stack.enter_async_context(
                    ClientSession(*transports[:2])
                )

            await session.initialize()
            self._session = session
            return session

        except Exception:
            # If session creation fails, clean up the exit stack
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            raise

    async def close(self):
        """Closes the session and cleans up resources."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                # Log the error but don't re-raise to avoid blocking shutdown
                print(
                    f"Warning: Error during MCP session cleanup: {e}", file=self._errlog
                )
            finally:
                self._exit_stack = None
                self._session = None


class CustomMCPToolset(MCPToolset):
    """
    Custom MCP Toolset that uses the CustomMcpSessionManager.

    This class replaces the default MCPToolset to enable the use of our
    custom session manager with configurable timeouts.
    """

    def __init__(
        self,
        connection_params: Union[StdioServerParameters, StreamableHTTPServerParams],
        tool_filter: Union[ToolPredicate, List[str], None] = None,
        errlog: TextIO = sys.stderr,
    ):
        # MCPToolset의 __init__을 반드시 호출
        super().__init__(
            connection_params=connection_params,
            tool_filter=tool_filter,
            errlog=errlog,
        )
        # 커스텀 세션 매니저로 교체
        self._mcp_session_manager = CustomMcpSessionManager(
            connection_params, errlog=errlog
        )
