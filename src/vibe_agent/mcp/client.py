from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from vibe_agent.config import MCPServerStdio, MCPServerSSE

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, server_config: MCPServerStdio | MCPServerSSE, timeout: float = 30.0) -> None:
        self._config = server_config
        self._timeout = timeout
        self._session: ClientSession | None = None
        self._cm_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        try:
            if isinstance(self._config, MCPServerStdio):
                params = StdioServerParameters(
                    command=self._config.command,
                    args=self._config.args,
                    env=self._config.env,
                )
                read, write = await self._enter(stdio_client(params))
            else:
                read, write, _ = await self._enter(
                    streamable_http_client(self._config.url, headers=self._config.headers or {})
                )

            self._session = await self._enter(ClientSession(read, write))
            await self._session.initialize()
            logger.info("Connected to MCP server: %s", self._config)
        except Exception:
            logger.warning("Failed to connect to MCP server: %s", self._config, exc_info=True)
            await self.disconnect()
            raise

    async def _enter(self, cm: Any) -> Any:
        if self._cm_stack is None:
            self._cm_stack = AsyncExitStack()
        return await self._cm_stack.enter_async_context(cm)

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("MCP session not connected")
        result = await asyncio.wait_for(
            self._session.call_tool(name, arguments=arguments),
            timeout=self._timeout,
        )
        if result.isError:
            parts = [str(c) for c in result.content]
            raise RuntimeError(f"MCP tool '{name}' error: {'; '.join(parts)}")
        parts = [str(c) for c in result.content]
        return "\n".join(parts)

    async def disconnect(self) -> None:
        if self._cm_stack is not None:
            await self._cm_stack.aclose()
            self._cm_stack = None
            self._session = None
