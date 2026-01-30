import json

import httpx
import pytest
import respx
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from mcphero.adapters.openai import MCPToolAdapterOpenAI


class TestGetToolDefinitions:
    @respx.mock
    async def test_converts_mcp_tools_to_openai_format(
        self, base_url, sample_mcp_tools
    ):
        respx.get(f"{base_url}/tools").mock(
            return_value=httpx.Response(200, json=sample_mcp_tools)
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tools = await adapter.get_tool_definitions()

        assert len(tools) == 2
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["function"]["description"] == "Get the current weather for a location"
        assert tools[0]["function"]["parameters"] == sample_mcp_tools[0]["inputSchema"]

    @respx.mock
    async def test_handles_missing_input_schema(
        self, base_url, sample_mcp_tool_no_schema
    ):
        respx.get(f"{base_url}/tools").mock(
            return_value=httpx.Response(200, json=sample_mcp_tool_no_schema)
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tools = await adapter.get_tool_definitions()

        assert len(tools) == 1
        assert tools[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    @respx.mock
    async def test_handles_empty_tools(self, base_url):
        respx.get(f"{base_url}/tools").mock(
            return_value=httpx.Response(200, json=[])
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tools = await adapter.get_tool_definitions()
        assert tools == []


class TestProcessToolCalls:
    @respx.mock
    async def test_success(self, base_url, sample_tool_result):
        respx.post(f"{base_url}/tools/get_weather/call").mock(
            return_value=httpx.Response(200, json=sample_tool_result)
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments=json.dumps({"location": "London"}),
                ),
            )
        ]

        results = await adapter.process_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert json.loads(results[0]["content"]) == sample_tool_result

    @respx.mock
    async def test_multiple_calls(self, base_url):
        respx.post(f"{base_url}/tools/get_weather/call").mock(
            return_value=httpx.Response(200, json={"temp": 72})
        )
        respx.post(f"{base_url}/tools/search/call").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments=json.dumps({"location": "London"}),
                ),
            ),
            ChatCompletionMessageToolCall(
                id="call_2",
                type="function",
                function=Function(
                    name="search",
                    arguments=json.dumps({"query": "test"}),
                ),
            ),
        ]

        results = await adapter.process_tool_calls(tool_calls)
        assert len(results) == 2
        assert results[0]["tool_call_id"] == "call_1"
        assert results[1]["tool_call_id"] == "call_2"

    @respx.mock
    async def test_invalid_json_arguments(self, base_url):
        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments="not valid json{{{",
                ),
            )
        ]

        results = await adapter.process_tool_calls(tool_calls)
        assert len(results) == 1
        content = json.loads(results[0]["content"])
        assert "error" in content
        assert "parse" in content["error"].lower()

    @respx.mock
    async def test_http_error(self, base_url):
        respx.post(f"{base_url}/tools/get_weather/call").mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments=json.dumps({"location": "London"}),
                ),
            )
        ]

        results = await adapter.process_tool_calls(tool_calls)
        assert len(results) == 1
        content = json.loads(results[0]["content"])
        assert "error" in content

    @respx.mock
    async def test_return_errors_false_omits_failures(self, base_url):
        respx.post(f"{base_url}/tools/get_weather/call").mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments=json.dumps({"location": "London"}),
                ),
            )
        ]

        results = await adapter.process_tool_calls(tool_calls, return_errors=False)
        assert len(results) == 0

    @respx.mock
    async def test_string_result_not_double_encoded(self, base_url):
        respx.post(f"{base_url}/tools/get_weather/call").mock(
            return_value=httpx.Response(200, json="plain string result")
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments=json.dumps({"location": "London"}),
                ),
            )
        ]

        results = await adapter.process_tool_calls(tool_calls)
        assert len(results) == 1
        # String results are passed through directly, not JSON-encoded again
        assert results[0]["content"] == "plain string result"

    @respx.mock
    async def test_dict_result_is_json_encoded(self, base_url, sample_tool_result):
        respx.post(f"{base_url}/tools/get_weather/call").mock(
            return_value=httpx.Response(200, json=sample_tool_result)
        )

        adapter = MCPToolAdapterOpenAI(base_url)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="get_weather",
                    arguments=json.dumps({"location": "London"}),
                ),
            )
        ]

        results = await adapter.process_tool_calls(tool_calls)
        assert json.loads(results[0]["content"]) == sample_tool_result
