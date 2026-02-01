import httpx
import pytest
import respx

try:
    from google.genai import types

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

from mcphero.adapters.base_adapter import MCPServerConfig

pytestmark = pytest.mark.skipif(
    not HAS_GOOGLE_GENAI, reason="google-genai not installed"
)


def _tools_response(tools):
    return {"jsonrpc": "2.0", "id": "1", "result": {"tools": tools}}


@pytest.fixture
def adapter(base_url):
    from mcphero.adapters.gemini import MCPToolAdapterGemini

    return MCPToolAdapterGemini(MCPServerConfig(url=base_url, init_mode="none"))


class TestGetFunctionDeclarations:
    @respx.mock
    async def test_converts_mcp_tools(self, base_url, adapter, sample_mcp_tools):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )

        declarations = await adapter.get_function_declarations()

        assert len(declarations) == 2
        assert declarations[0].name == "get_weather"
        assert declarations[0].description == "Get the current weather for a location"

    @respx.mock
    async def test_handles_missing_schema(
        self, base_url, adapter, sample_mcp_tool_no_schema
    ):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tool_no_schema)
            )
        )

        declarations = await adapter.get_function_declarations()

        assert len(declarations) == 1
        assert declarations[0].name == "ping"

    @respx.mock
    async def test_empty_list(self, base_url, adapter):
        respx.post(base_url).mock(
            return_value=httpx.Response(200, json=_tools_response([]))
        )

        declarations = await adapter.get_function_declarations()
        assert declarations == []


class TestGetTool:
    @respx.mock
    async def test_returns_tool_wrapping_declarations(
        self, base_url, adapter, sample_mcp_tools
    ):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )

        tool = await adapter.get_tool()

        assert isinstance(tool, types.Tool)
        assert len(tool.function_declarations) == 2


class TestProcessFunctionCalls:
    @respx.mock
    async def test_success(self, base_url, adapter, sample_mcp_tools, sample_tool_result):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_tools_response(sample_mcp_tools)),
                httpx.Response(200, json=sample_tool_result),
            ]
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        results = await adapter.process_function_calls([fc])

        assert len(results) == 1
        assert isinstance(results[0], types.Content)
        assert results[0].role == "user"
        assert len(results[0].parts) == 1

    @respx.mock
    async def test_non_dict_result_wrapped(self, base_url, adapter, sample_mcp_tools):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_tools_response(sample_mcp_tools)),
                httpx.Response(200, json="string result"),
            ]
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        results = await adapter.process_function_calls([fc])

        assert len(results) == 1

    @respx.mock
    async def test_dict_result_passed_directly(self, base_url, adapter, sample_mcp_tools):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_tools_response(sample_mcp_tools)),
                httpx.Response(200, json={"temp": 72}),
            ]
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        results = await adapter.process_function_calls([fc])

        assert len(results) == 1

    @respx.mock
    async def test_http_error(self, base_url, adapter, sample_mcp_tools):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )
        await adapter.discover_tools()

        respx.reset()
        respx.post(base_url).mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        results = await adapter.process_function_calls([fc])

        assert len(results) == 1

    @respx.mock
    async def test_return_errors_false(self, base_url, adapter, sample_mcp_tools):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )
        await adapter.discover_tools()

        respx.reset()
        respx.post(base_url).mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        results = await adapter.process_function_calls([fc], return_errors=False)

        assert len(results) == 0

    @respx.mock
    async def test_empty_args(self, base_url, adapter, sample_mcp_tool_no_schema):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_tools_response(sample_mcp_tool_no_schema)),
                httpx.Response(200, json={"status": "ok"}),
            ]
        )

        fc = types.FunctionCall(name="ping", args=None)
        results = await adapter.process_function_calls([fc])

        assert len(results) == 1


class TestProcessFunctionCallsAsParts:
    @respx.mock
    async def test_returns_parts(self, base_url, adapter, sample_mcp_tools, sample_tool_result):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )
        await adapter.discover_tools()

        respx.reset()
        respx.post(base_url).mock(
            return_value=httpx.Response(200, json=sample_tool_result)
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        parts = await adapter.process_function_calls_as_parts([fc])

        assert len(parts) == 1
        assert isinstance(parts[0], types.Part)

    @respx.mock
    async def test_error_returns_part(self, base_url, adapter, sample_mcp_tools):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )
        await adapter.discover_tools()

        respx.reset()
        respx.post(base_url).mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        fc = types.FunctionCall(name="get_weather", args={"location": "London"})
        parts = await adapter.process_function_calls_as_parts([fc])

        assert len(parts) == 1
        assert isinstance(parts[0], types.Part)


class TestCreateFunctionResponseContent:
    def test_returns_correct_content(self):
        from mcphero.adapters.gemini import MCPToolAdapterGemini

        content = MCPToolAdapterGemini.create_function_response_content(
            name="get_weather",
            response={"temp": 72},
        )

        assert isinstance(content, types.Content)
        assert content.role == "user"
        assert len(content.parts) == 1
