# mcphero/adapters/base_adapter.py
"""
Base MCP adapter supporting 1-N servers.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypedDict, TypeVar
from urllib.parse import urlparse

import httpx

from mcphero.__about__ import __version__

PROTOCOL_VERSION = "2025-06-18"


class JsonRpcRequest(TypedDict):
    id: str
    jsonrpc: str
    method: str
    params: dict[str, Any]


class JsonRpcNotification(TypedDict):
    jsonrpc: str
    method: str


class _JsonRpcResponseRequired(TypedDict):
    jsonrpc: str
    id: str


class JsonRpcResponse(_JsonRpcResponseRequired, total=False):
    result: dict[str, Any]
    error: dict[str, Any]


class _RawMCPToolRequired(TypedDict):
    name: str


class RawMCPTool(_RawMCPToolRequired, total=False):
    description: str
    inputSchema: dict[str, Any]


class InitMode(Enum):
    auto = "auto"
    on_fail = "on_fail"
    none = "none"


@dataclass
class MCPServerConfig:
    """
    Configuration for a single MCP server.
    It's important to specify AUTH headers as most MCP servers force auth nowadays.
    """

    url: str
    name: str | None = None
    timeout: float = 30.0
    headers: dict[str, str] | None = None
    init_mode: InitMode | str = InitMode.auto
    tool_prefix: str | None = None

    def __post_init__(self):
        if self.name is None:
            parsed = urlparse(self.url)
            hostname = parsed.hostname or "unknown"
            parts = hostname.split(".")
            # Take the second-level domain part (e.g. "context7" from mcp.context7.com,
            # "mcphero" from api.mcphero.app)
            domain = parts[-2] if len(parts) >= 2 else hostname
            path_part = parsed.path.rstrip("/").split("/")[-1] or "server"
            self.name = f"{domain}_{path_part}"
        if isinstance(self.init_mode, str):
            self.init_mode = InitMode(self.init_mode)


@dataclass
class MCPToolDefinition:
    """A discovered MCP tool with routing information."""

    name: str  # The name exposed to the LLM (possibly prefixed)
    original_name: str  # The original name on the MCP server
    server_name: str  # Which server this tool belongs to
    description: str
    input_schema: dict[str, Any]

    # Keep reference to raw tool if needed
    raw: RawMCPTool = field(repr=False)


@dataclass
class ToolMapping:
    """Internal mapping for routing tool calls."""

    server_name: str
    original_name: str
    prefixed_name: str
    connection: MCPConnection


class MCPConnection:
    """
    Single MCP server connection handler.
    Handles the entire lifecycle from initialization to independent requests.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session_id: str | None = None
        self._initialize_result: JsonRpcResponse | None = None
        self._protocol_version: str | None = None

    @staticmethod
    def _parse_response(response: httpx.Response) -> JsonRpcResponse:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return MCPConnection._parse_sse_response(response.text)
        return response.json()

    @staticmethod
    def _parse_sse_response(text: str) -> JsonRpcResponse:
        data_lines: list[str] = []
        result = None
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
            elif line == "":
                if data_lines:
                    result = json.loads("\n".join(data_lines))
                    data_lines = []
        if data_lines:
            result = json.loads("\n".join(data_lines))
        if result is None:
            raise ValueError("No data field found in SSE response")
        return result

    async def initialize(self) -> JsonRpcResponse:
        if self._initialize_result is not None:
            return self._initialize_result

        init_payload = {
            "id": str(uuid.uuid4()),
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mcphero", "version": __version__},
            },
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            **(self.config.headers or {}),
        }

        async with httpx.AsyncClient(
            timeout=self.config.timeout, headers=headers, follow_redirects=True
        ) as client:
            response = await client.post(self.config.url, json=init_payload)
            response.raise_for_status()

            if session_id := response.headers.get("Mcp-Session-Id"):
                self._session_id = session_id

            result = self._parse_response(response)
            self._protocol_version = result.get("result", {}).get(
                "protocolVersion", PROTOCOL_VERSION
            )
            self._initialize_result = result

            notification: JsonRpcNotification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            notify_headers = dict(headers)
            if self._session_id:
                notify_headers["Mcp-Session-Id"] = self._session_id
            await client.post(
                self.config.url, json=notification, headers=notify_headers
            )

        return result

    async def _ensure_initialized(self) -> None:
        if self.config.init_mode == InitMode.auto:
            await self.initialize()

    async def _make_request(
        self, data: JsonRpcRequest, *, _retry: bool = True
    ) -> JsonRpcResponse:
        headers = {
            "Accept": "application/json, text/event-stream",
            **(self.config.headers or {}),
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version

        async with httpx.AsyncClient(
            timeout=self.config.timeout, headers=headers, follow_redirects=True
        ) as client:
            response = await client.post(self.config.url, json=data)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                if (
                    _retry
                    and self.config.init_mode == InitMode.on_fail
                    and self._initialize_result is None
                ):
                    await self.initialize()
                    return await self._make_request(data, _retry=False)
                raise
            return self._parse_response(response)

    async def get_tools(self) -> list[RawMCPTool]:
        await self._ensure_initialized()
        result = await self._make_request(
            {
                "id": str(uuid.uuid4()),
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
            }
        )
        return result.get("result", {}).get("tools", [])

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> JsonRpcResponse:
        await self._ensure_initialized()
        return await self._make_request(
            {
                "id": str(uuid.uuid4()),
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )


ToolCallT = TypeVar("ToolCallT")
ResultT = TypeVar("ResultT")


class BaseAdapter(Generic[ToolCallT, ResultT]):
    """
    Base MCP adapter supporting one or many servers.

    Args:
        servers: Single URL/config or list of URLs/configs.
        auto_prefix_on_collision: Auto-prefix tools when names collide across servers.
        prefix_separator: Separator for prefixed names (default: "__").
    """

    def __init__(
        self,
        servers: str | MCPServerConfig | list[str | MCPServerConfig],
        *,
        auto_prefix_on_collision: bool = True,
        prefix_separator: str = "__",
    ):
        if isinstance(servers, (str, MCPServerConfig)):
            servers = [servers]

        self._configs: list[MCPServerConfig] = [
            MCPServerConfig(url=s) if isinstance(s, str) else s for s in servers
        ]
        self.auto_prefix_on_collision = auto_prefix_on_collision
        self.prefix_separator = prefix_separator

        self._connections: dict[str, MCPConnection] = {}
        self._prepare_connections()
        self._tool_map: dict[str, ToolMapping] = {}
        self._tools: list[MCPToolDefinition] = []
        self._discovered = False

    def _prepare_connections(self):
        seen_names: set[str] = set()
        for cfg in self._configs:
            assert cfg.name is not None
            base_name = cfg.name
            unique_name = base_name
            counter = 1

            while unique_name in seen_names:
                if not self.auto_prefix_on_collision:
                    raise ValueError(
                        f"Duplicate server name '{base_name}'. "
                        "Provide unique names explicitly or enable auto_prefix_on_collision."
                    )
                counter += 1
                unique_name = f"{base_name}{self.prefix_separator}{counter}"

            seen_names.add(unique_name)
            if unique_name != base_name:
                cfg.name = unique_name  # Keep config in sync for later look-ups

            self._connections[unique_name] = MCPConnection(cfg)

    @property
    def is_multi_server(self) -> bool:
        return len(self._configs) > 1

    @property
    def tools(self) -> list[MCPToolDefinition]:
        """Discovered tools. Call discover_tools() first."""
        if not self._discovered:
            raise RuntimeError("Must call discover_tools() first")
        return self._tools

    def _make_prefixed_name(self, prefix: str | None, tool_name: str) -> str:
        if prefix:
            return f"{prefix}{self.prefix_separator}{tool_name}"
        return tool_name

    def _resolve_tools(
        self, tools_by_server: dict[str, list[RawMCPTool]]
    ) -> list[MCPToolDefinition]:
        """Resolve naming collisions and build typed tool definitions."""
        # Find collisions
        tool_sources: dict[str, list[str]] = {}
        for server_name, tools in tools_by_server.items():
            config = next(c for c in self._configs if c.name == server_name)
            for tool in tools:
                name = self._make_prefixed_name(config.tool_prefix, tool["name"])
                tool_sources.setdefault(name, []).append(server_name)

        collisions = {
            name for name, sources in tool_sources.items() if len(sources) > 1
        }

        # Build definitions
        definitions: list[MCPToolDefinition] = []
        self._tool_map.clear()

        for server_name, tools in tools_by_server.items():
            config = next(c for c in self._configs if c.name == server_name)
            conn = self._connections[server_name]

            for tool in tools:
                base_name = self._make_prefixed_name(config.tool_prefix, tool["name"])

                if base_name in collisions and self.auto_prefix_on_collision:
                    final_name = f"{server_name}{self.prefix_separator}{tool['name']}"
                else:
                    final_name = base_name

                definition = MCPToolDefinition(
                    name=final_name,
                    original_name=tool["name"],
                    server_name=server_name,
                    description=tool.get("description", ""),
                    input_schema=tool.get(
                        "inputSchema", {"type": "object", "properties": {}}
                    ),
                    raw=tool,
                )
                definitions.append(definition)

                self._tool_map[final_name] = ToolMapping(
                    server_name=server_name,
                    original_name=tool["name"],
                    prefixed_name=final_name,
                    connection=conn,
                )

        return definitions

    async def discover_tools(self, *, parallel: bool = True) -> list[MCPToolDefinition]:
        """
        Discover tools from all servers.

        Returns:
            List of typed tool definitions.
        """
        if parallel and self.is_multi_server:

            async def fetch_one(name: str) -> tuple[str, list[RawMCPTool]]:
                return name, await self._connections[name].get_tools()

            results = await asyncio.gather(
                *[fetch_one(name) for name in self._connections],
                return_exceptions=True,
            )
            tools_by_server: dict[str, list[RawMCPTool]] = {}
            for result in results:
                if not isinstance(result, BaseException):
                    name, tools = result
                    tools_by_server[name] = tools
        else:
            tools_by_server = {}
            for name, conn in self._connections.items():
                try:
                    tools_by_server[name] = await conn.get_tools()
                except Exception:
                    continue

        self._tools = self._resolve_tools(tools_by_server)
        self._discovered = True
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> JsonRpcResponse:
        """Call a tool by its name (as exposed to the LLM)."""
        if not self._discovered:
            await self.discover_tools()

        mapping = self._tool_map.get(name)
        if mapping is None:
            raise KeyError(f"Unknown tool: {name}")

        return await mapping.connection.call_tool(mapping.original_name, arguments)

    async def _execute_single_tool_call(
        self, tool_call: ToolCallT, *, return_errors: bool
    ) -> ResultT | None:
        """Execute a single provider-specific tool call.

        Each child adapter must implement this. It receives one provider-specific
        tool-call object and returns one provider-specific result (or ``None`` to skip).
        """
        raise NotImplementedError

    async def process_tool_calls(
        self,
        tool_calls: list[ToolCallT],
        *,
        return_errors: bool = True,
        parallel: bool = True,
    ) -> list[ResultT]:
        """Generic loop: run ``_execute_single_tool_call`` for every item.

        Handles early-return on empty list, parallel/sequential execution,
        and filters out ``None`` results.
        """
        if not tool_calls:
            return []

        if parallel:
            results = await asyncio.gather(
                *[
                    self._execute_single_tool_call(tc, return_errors=return_errors)
                    for tc in tool_calls
                ],
                return_exceptions=True,
            )
            return [
                r for r in results if r is not None and not isinstance(r, BaseException)
            ]
        else:
            return [
                r
                for tc in tool_calls
                if (
                    r := await self._execute_single_tool_call(
                        tc, return_errors=return_errors
                    )
                )
                is not None
            ]

    async def initialize_all(
        self, *, parallel: bool = True
    ) -> dict[str, JsonRpcResponse | Exception]:
        """Pre-initialize all connections."""

        async def init_one(name: str) -> tuple[str, JsonRpcResponse | Exception]:
            try:
                return name, await self._connections[name].initialize()
            except Exception as e:
                return name, e

        if parallel:
            results = await asyncio.gather(*[init_one(n) for n in self._connections])
            return dict(results)
        else:
            return {name: (await init_one(name))[1] for name in self._connections}
