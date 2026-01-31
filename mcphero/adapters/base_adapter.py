from __future__ import annotations

import json
import uuid
from enum import Enum

import httpx

from mcphero.__about__ import __version__

PROTOCOL_VERSION = "2025-06-18"


class InitMode(Enum):
    auto = "auto"
    on_fail = "on_fail"
    none = "none"


class BaseAdapter:
    """Async HTTP adapter for communicating with an MCP server over Streamable HTTP.

    Handles the MCP session lifecycle (initialize + notifications/initialized)
    and provides methods for listing and calling tools via JSON-RPC.

    Args:
        base_url: Root URL of the MCP server endpoint.
        timeout: HTTP request timeout in seconds.
        headers: Extra headers merged into every request.
        init_mode: Controls when the MCP session is initialized.
            - "auto"   -- initialize before the first request (default).
            - "on_fail" -- skip init upfront; if a request fails with an
              HTTP error and the session hasn't been initialized yet,
              initialize and retry the request once.
            - "none"    -- never auto-initialize; errors propagate as-is.
            Accepts an :class:`InitMode` enum member or its string value.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict | None = None,
        init_mode: InitMode | str = InitMode.auto,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        self.init_mode = (
            InitMode(init_mode) if isinstance(init_mode, str) else init_mode
        )
        self._session_id: str | None = None
        self._initialize_result: dict | None = None
        self._protocol_version: str | None = None

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict:
        """Parse an HTTP response as JSON or SSE based on Content-Type."""
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return BaseAdapter._parse_sse_response(response.text)
        return response.json()

    @staticmethod
    def _parse_sse_response(text: str) -> dict:
        """Extract the last JSON-RPC message from an SSE stream."""
        data_lines: list[str] = []
        result = None
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
            elif line == "":
                # blank line = end of event
                if data_lines:
                    result = json.loads("\n".join(data_lines))
                    data_lines = []
        # handle stream that doesn't end with a trailing blank line
        if data_lines:
            result = json.loads("\n".join(data_lines))
        if result is None:
            raise ValueError("No data field found in SSE response")
        return result

    async def initialize(self) -> dict:
        if self._initialize_result is not None:
            return self._initialize_result

        # Step 1: Send initialize JSON-RPC request
        init_payload = {
            "id": str(uuid.uuid4()),
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "mcphero",
                    "version": __version__,
                },
            },
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.post(self.base_url, json=init_payload)
            response.raise_for_status()

            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id

            result = self._parse_response(response)
            self._protocol_version = result.get("result", {}).get(
                "protocolVersion", PROTOCOL_VERSION
            )
            self._initialize_result = result

            # Step 2: Send notifications/initialized notification
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }

            notify_headers = dict(headers)
            if self._session_id:
                notify_headers["Mcp-Session-Id"] = self._session_id

            await client.post(self.base_url, json=notification, headers=notify_headers)

        return result

    async def _ensure_initialized(self) -> None:
        if self.init_mode == InitMode.auto:
            await self.initialize()

    async def _make_request(
        self,
        data: dict,
        *,
        _retry: bool = True,
    ) -> dict:
        headers = {
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.post(self.base_url, json=data)

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                if (
                    _retry
                    and self.init_mode == InitMode.on_fail
                    and self._initialize_result is None
                ):
                    await self.initialize()
                    return await self._make_request(data, _retry=False)
                raise
            return self._parse_response(response)

    async def get_mcp_tools(self) -> dict:
        await self._ensure_initialized()
        return await self._make_request(
            {
                "id": str(uuid.uuid4()),
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
            }
        )

    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> dict:
        await self._ensure_initialized()
        return await self._make_request(
            {
                "id": str(uuid.uuid4()),
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        )
