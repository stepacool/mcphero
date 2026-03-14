# mcphero/adapters/generic.py
"""Provider-agnostic MCP adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from mcphero.adapters.base_adapter import BaseAdapter


@dataclass
class GenericToolCall:
    """A provider-agnostic tool call."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class GenericToolResult:
    """A provider-agnostic tool result."""

    id: str
    content: Any  # Raw result (dict/list/str) from the MCP server
    is_error: bool = False


class MCPToolAdapter(BaseAdapter[GenericToolCall, GenericToolResult]):
    """
    Provider-agnostic MCP adapter. Works without any LLM SDK dependency.

    Use this when your framework has its own tool-call loop and you just need
    to execute MCP tools and get results back.

    Usage:
        adapter = MCPToolAdapter("https://mcp.example.com/server")

        tools = await adapter.discover_tools()  # list[MCPToolDefinition]

        results = await adapter.process_tool_calls([
            GenericToolCall(name="tool_name", arguments={"key": "value"}, id="1"),
        ])
        # results: list[GenericToolResult]

    Or call a single tool directly (no wrapping needed):
        result = await adapter.call_tool("tool_name", {"key": "value"})
    """

    async def _execute_single_tool_call(
        self, tool_call: GenericToolCall, *, return_errors: bool
    ) -> GenericToolResult | None:
        try:
            response = await self.call_tool(tool_call.name, tool_call.arguments)
            return GenericToolResult(id=tool_call.id, content=response)
        except (KeyError, httpx.HTTPError, Exception) as e:
            if return_errors:
                return GenericToolResult(
                    id=tool_call.id,
                    content={"error": str(e)},
                    is_error=True,
                )
            return None
