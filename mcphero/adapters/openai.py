# mcphero/adapters/openai.py
"""MCP Tool Adapter for OpenAI."""

from __future__ import annotations

import json

import httpx
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)

from mcphero.adapters.base_adapter import BaseAdapter, MCPToolDefinition


class MCPToolAdapterOpenAI(
    BaseAdapter[ChatCompletionMessageToolCall, ChatCompletionToolMessageParam]
):
    """
    Adapter for OpenAI. Supports single or multiple MCP servers.

    Usage:
        adapter = MCPToolAdapterOpenAI("https://mcp.example.com/server")
        # or
        adapter = MCPToolAdapterOpenAI([
            MCPServerConfig(url="https://mcp.example.com/weather", name="weather"),
            MCPServerConfig(url="https://mcp.example.com/calendar", name="calendar"),
        ])

        tools = await adapter.get_tool_definitions()
        response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

        if response.choices[0].message.tool_calls:
            results = await adapter.process_tool_calls(response.choices[0].message.tool_calls)
    """

    @staticmethod
    def _to_openai_tool(tool: MCPToolDefinition) -> ChatCompletionToolParam:
        """Convert MCPToolDefinition to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    async def get_tool_definitions(
        self, *, parallel: bool = True
    ) -> list[ChatCompletionToolParam]:
        """Fetch tools and return OpenAI-compatible definitions."""
        tools = await self.discover_tools(parallel=parallel)
        return [self._to_openai_tool(t) for t in tools]

    async def _execute_single_tool_call(
        self, tool_call: ChatCompletionMessageToolCall, *, return_errors: bool
    ) -> ChatCompletionToolMessageParam | None:
        tc = tool_call
        try:
            arguments = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            if return_errors:
                return {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": json.dumps({"error": "Failed to parse arguments"}),
                }
            return None

        try:
            response = await self.call_tool(tc.function.name, arguments)
            content = (
                json.dumps(response) if not isinstance(response, str) else response
            )
            return {
                "tool_call_id": tc.id,
                "role": "tool",
                "content": content,
            }
        except (KeyError, httpx.HTTPError, Exception) as e:
            if return_errors:
                return {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": json.dumps({"error": str(e)}),
                }
            return None
