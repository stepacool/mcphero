"""
MCP Tool Adapter for OpenAI.

This adapter converts remote MCP server tools to OpenAI-compatible tool definitions
and processes OpenAI's tool calls to HTTP requests to the MCP server.
"""

from __future__ import annotations

import json

import httpx
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)

from mcphero.adapters.base_adapter import BaseAdapter


class MCPToolAdapterOpenAI(BaseAdapter):
    """
    Adapter that converts remote MCP server tools to OpenAI-compatible tool definitions.
    Also converts OpenAI's tool_calls to HTTP requests to the MCP server.

    Usage:

        from openai import OpenAI
        from mcphero import MCPToolAdapterOpenAI

        adapter = MCPToolAdapterOpenAI("https://api.mcphero.app/mcp/your-server-id")
        client = OpenAI()

        # Get tool definitions
        tools = await adapter.get_tool_definitions()

        # Make request with tools
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tools,
        )

        # Process tool calls if present
        if response.choices[0].message.tool_calls:
            tool_results = await adapter.process_tool_calls(
                response.choices[0].message.tool_calls
            )

            # Continue conversation with results
            messages = [
                {"role": "user", "content": "What's the weather?"},
                response.choices[0].message,
                *tool_results,
            ]
            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
    """

    async def get_tool_definitions(self) -> list[ChatCompletionToolParam]:
        """
        Fetch tools from MCP server and convert them to OpenAI tool schemas.

        Returns:
            List of tool definitions compatible with OpenAI's `tools` parameter.

        Example:
            tools = await adapter.get_tool_definitions()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
        """
        mcp_tools = await self.get_mcp_tools()

        openai_tools: list[ChatCompletionToolParam] = []
        for tool in mcp_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "inputSchema",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    },
                }
            )

        return openai_tools

    async def process_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        return_errors: bool = True,
    ) -> list[ChatCompletionToolMessageParam]:
        """
        Process OpenAI's tool_calls by invoking the tools via HTTP.

        Args:
            tool_calls: List of tool calls from `response.choices[0].message.tool_calls`.
            return_errors: If True, include error messages for failed calls.
                If False, failed calls are omitted from results.

        Returns:
            List of tool message dicts compatible with OpenAI's messages format.
            Each result has `role="tool"`, `tool_call_id`, and `content`.

        Example:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )

            if response.choices[0].message.tool_calls:
                tool_results = await adapter.process_tool_calls(
                    response.choices[0].message.tool_calls
                )

                # Add to conversation
                messages.append(response.choices[0].message)
                messages.extend(tool_results)
        """
        results: list[ChatCompletionToolMessageParam] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                if return_errors:
                    results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": json.dumps(
                                {"error": "Failed to parse tool arguments"}
                            ),
                        }
                    )
                continue

            try:
                result = await self.call_mcp_tool(tool_name, arguments)

                results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": json.dumps(result)
                        if not isinstance(result, str)
                        else result,
                    }
                )

            except httpx.HTTPError as e:
                if return_errors:
                    results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": json.dumps(
                                {"error": f"HTTP error calling tool: {str(e)}"}
                            ),
                        }
                    )

            except Exception as e:
                if return_errors:
                    results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": json.dumps(
                                {"error": f"Unexpected error: {str(e)}"}
                            ),
                        }
                    )

        return results
