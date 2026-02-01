# mcphero/adapters/openai.py
"""MCP Tool Adapter for OpenAI."""

from __future__ import annotations

import asyncio
import json

import httpx
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)

from mcphero.adapters.base_adapter import BaseAdapter, MCPToolDefinition


class MCPToolAdapterOpenAI(BaseAdapter):
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

    async def get_tool_definitions(self, *, parallel: bool = True) -> list[ChatCompletionToolParam]:
        """Fetch tools and return OpenAI-compatible definitions."""
        tools = await self.discover_tools(parallel=parallel)
        return [self._to_openai_tool(t) for t in tools]

    async def process_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        *,
        return_errors: bool = True,
        parallel: bool = True,
    ) -> list[ChatCompletionToolMessageParam]:
        """
        Process OpenAI tool calls, routing to appropriate servers.
        """
        if not tool_calls:
            return []

        async def execute_one(tc: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam | None:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                if return_errors:
                    result: ChatCompletionToolMessageParam =  {
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "content": json.dumps({"error": "Failed to parse arguments"}),
                    }
                    return result
                return None

            try:
                response = await self.call_tool(tc.function.name, arguments)
                content = json.dumps(response) if not isinstance(response, str) else response
                result: ChatCompletionToolMessageParam = {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": content,
                }
                return result
            except (KeyError, httpx.HTTPError, Exception) as e:
                if return_errors:
                    result: ChatCompletionToolMessageParam =  {
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "content": json.dumps({"error": str(e)}),
                    }
                    return result
                return None

        if parallel:
            results = await asyncio.gather(*[execute_one(tc) for tc in tool_calls], return_exceptions=True)
            return [r for r in results if r is not None and not isinstance(r, BaseException)]
        else:
            return [r for tc in tool_calls if (r := await execute_one(tc)) is not None]
