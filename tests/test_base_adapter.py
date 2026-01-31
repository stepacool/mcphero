import json

import httpx
import pytest
import respx

from mcphero.__about__ import __version__
from mcphero.adapters.base_adapter import PROTOCOL_VERSION, BaseAdapter, InitMode


def _sse_body(payload: dict) -> str:
    """Format a dict as an SSE event body."""
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


class TestBaseAdapterInit:
    def test_stores_base_url(self, base_url):
        adapter = BaseAdapter(base_url)
        assert adapter.base_url == base_url

    def test_strips_trailing_slash(self):
        adapter = BaseAdapter("https://example.com/")
        assert adapter.base_url == "https://example.com"

    def test_default_timeout(self, base_url):
        adapter = BaseAdapter(base_url)
        assert adapter.timeout == 30.0

    def test_custom_timeout(self, base_url):
        adapter = BaseAdapter(base_url, timeout=60.0)
        assert adapter.timeout == 60.0

    def test_default_headers(self, base_url):
        adapter = BaseAdapter(base_url)
        assert adapter.headers == {}

    def test_custom_headers(self, base_url):
        headers = {"Authorization": "Bearer token"}
        adapter = BaseAdapter(base_url, headers=headers)
        assert adapter.headers == headers

    def test_initial_session_state(self, base_url):
        adapter = BaseAdapter(base_url)
        assert adapter._session_id is None
        assert adapter._initialize_result is None
        assert adapter._protocol_version is None

    def test_default_init_mode_is_auto(self, base_url):
        adapter = BaseAdapter(base_url)
        assert adapter.init_mode == InitMode.auto

    def test_accepts_string_init_mode(self, base_url):
        adapter = BaseAdapter(base_url, init_mode="on_fail")
        assert adapter.init_mode == InitMode.on_fail

    def test_accepts_enum_init_mode(self, base_url):
        adapter = BaseAdapter(base_url, init_mode=InitMode.none)
        assert adapter.init_mode == InitMode.none


class TestInitialize:
    @respx.mock
    async def test_sends_initialize_request(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "test-server", "version": "1.0"},
            },
        }
        init_route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "srv-session-123"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        result = await adapter.initialize()

        assert result == init_result
        assert init_route.call_count == 2

        # Verify the initialize request payload
        first_call = init_route.calls[0]
        body = first_call.request.content
        import json

        payload = json.loads(body)
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "initialize"
        assert payload["params"]["protocolVersion"] == PROTOCOL_VERSION
        assert payload["params"]["clientInfo"]["name"] == "mcphero"
        assert payload["params"]["clientInfo"]["version"] == __version__
        assert payload["params"]["capabilities"] == {}

    @respx.mock
    async def test_extracts_session_id_from_response(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "my-session-42"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        await adapter.initialize()

        assert adapter._session_id == "my-session-42"

    @respx.mock
    async def test_sends_initialized_notification(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "sess-1"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        await adapter.initialize()

        # Second call should be the notification
        import json

        second_call = route.calls[1]
        payload = json.loads(second_call.request.content)
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "notifications/initialized"
        assert "id" not in payload

    @respx.mock
    async def test_notification_includes_session_id_header(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "sess-abc"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        await adapter.initialize()

        second_call = route.calls[1]
        assert second_call.request.headers.get("Mcp-Session-Id") == "sess-abc"

    @respx.mock
    async def test_idempotent_returns_cached_result(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "sess-1"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        result1 = await adapter.initialize()
        result2 = await adapter.initialize()

        assert result1 == result2
        # Only 2 calls total (init + notification), not 4
        assert route.call_count == 2

    @respx.mock
    async def test_works_without_session_id(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        result = await adapter.initialize()

        assert result == init_result
        assert adapter._session_id is None

    @respx.mock
    async def test_stores_negotiated_protocol_version(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": "2024-11-05"},
        }
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        await adapter.initialize()

        assert adapter._protocol_version == "2024-11-05"


class TestMakeRequest:
    @respx.mock
    async def test_post_returns_parsed_json(self, base_url):
        expected = {"jsonrpc": "2.0", "id": "1", "result": {"tools": []}}
        respx.post(base_url).mock(
            return_value=httpx.Response(200, json=expected)
        )

        adapter = BaseAdapter(base_url)
        result = await adapter._make_request(
            {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )
        assert result == expected

    @respx.mock
    async def test_http_error_raises(self, base_url):
        respx.post(base_url).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        adapter = BaseAdapter(base_url)
        with pytest.raises(httpx.HTTPStatusError):
            await adapter._make_request(
                {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
            )

    @respx.mock
    async def test_no_session_id_header_before_init(self, base_url):
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {}})
        )

        adapter = BaseAdapter(base_url)
        await adapter._make_request(
            {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        request = route.calls[0].request
        assert "Mcp-Session-Id" not in request.headers

    @respx.mock
    async def test_includes_session_id_after_init(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "sess-xyz"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        await adapter.initialize()

        # Now mock a subsequent request
        respx.reset()
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": "2", "result": {}})
        )

        await adapter._make_request(
            {"id": "2", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        request = route.calls[0].request
        assert request.headers.get("Mcp-Session-Id") == "sess-xyz"

    @respx.mock
    async def test_includes_protocol_version_after_init(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": "2024-11-05"},
        }
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        await adapter.initialize()

        respx.reset()
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": "2", "result": {}})
        )

        await adapter._make_request(
            {"id": "2", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        request = route.calls[0].request
        assert request.headers.get("MCP-Protocol-Version") == "2024-11-05"


class TestInitModeAuto:
    @respx.mock
    async def test_get_mcp_tools_calls_initialize_before_request(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        tools_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"tools": []},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                # initialize request
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "s1"}),
                # notifications/initialized
                httpx.Response(202),
                # tools/list request
                httpx.Response(200, json=tools_result),
            ]
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.auto)
        result = await adapter.get_mcp_tools()

        assert result == tools_result
        # 3 calls: initialize, notification, tools/list
        assert route.call_count == 3

    @respx.mock
    async def test_call_mcp_tool_calls_initialize_before_request(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        call_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"content": [{"type": "text", "text": "hello"}]},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "s1"}),
                httpx.Response(202),
                httpx.Response(200, json=call_result),
            ]
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.auto)
        result = await adapter.call_mcp_tool("test_tool", {"arg": "val"})

        assert result == call_result
        assert route.call_count == 3


class TestInitModeOnFail:
    @respx.mock
    async def test_does_not_call_initialize_upfront(self, base_url):
        tools_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"tools": []},
        }
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json=tools_result),
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.on_fail)
        result = await adapter.get_mcp_tools()

        assert result == tools_result
        # Only 1 call: the tools/list request, no init
        assert route.call_count == 1

    @respx.mock
    async def test_initializes_and_retries_on_failure(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        tools_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"tools": []},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                # First tools/list fails
                httpx.Response(400, text="Bad Request"),
                # initialize request
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "s1"}),
                # notifications/initialized
                httpx.Response(202),
                # Retry tools/list succeeds
                httpx.Response(200, json=tools_result),
            ]
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.on_fail)
        result = await adapter.get_mcp_tools()

        assert result == tools_result
        # 4 calls: failed tools/list, init, notification, retry tools/list
        assert route.call_count == 4

    @respx.mock
    async def test_no_retry_if_already_initialized(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        # Pre-initialize the adapter
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "s1"}),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.on_fail)
        await adapter.initialize()

        respx.reset()
        # Now a request fails — should NOT retry since already initialized
        respx.post(base_url).mock(
            return_value=httpx.Response(400, text="Bad Request"),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await adapter.get_mcp_tools()

    @respx.mock
    async def test_error_propagates_if_retry_also_fails(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"protocolVersion": PROTOCOL_VERSION},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                # First tools/list fails
                httpx.Response(400, text="Bad Request"),
                # initialize succeeds
                httpx.Response(200, json=init_result, headers={"Mcp-Session-Id": "s1"}),
                # notifications/initialized
                httpx.Response(202),
                # Retry tools/list also fails
                httpx.Response(500, text="Internal Server Error"),
            ]
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.on_fail)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.get_mcp_tools()

        assert exc_info.value.response.status_code == 500
        assert route.call_count == 4


class TestInitModeNone:
    @respx.mock
    async def test_does_not_call_initialize(self, base_url):
        tools_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"tools": []},
        }
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json=tools_result),
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.none)
        result = await adapter.get_mcp_tools()

        assert result == tools_result
        # Only 1 call: tools/list, no init
        assert route.call_count == 1

    @respx.mock
    async def test_error_propagates_directly(self, base_url):
        route = respx.post(base_url).mock(
            return_value=httpx.Response(400, text="Bad Request"),
        )

        adapter = BaseAdapter(base_url, init_mode=InitMode.none)

        with pytest.raises(httpx.HTTPStatusError):
            await adapter.get_mcp_tools()

        # Only 1 call, no retry, no init
        assert route.call_count == 1


class TestParseSSE:
    def test_parses_json_content_type(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {}}
        response = httpx.Response(
            200,
            json=payload,
            headers={"content-type": "application/json"},
        )
        assert BaseAdapter._parse_response(response) == payload

    def test_parses_sse_content_type(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {"tools": []}}
        sse_body = _sse_body(payload)
        response = httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )
        assert BaseAdapter._parse_response(response) == payload

    def test_parses_sse_without_trailing_blank_line(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {}}
        sse_body = f"event: message\ndata: {json.dumps(payload)}"
        response = httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )
        assert BaseAdapter._parse_response(response) == payload

    def test_parses_sse_with_charset(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {}}
        sse_body = _sse_body(payload)
        response = httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
        assert BaseAdapter._parse_response(response) == payload

    def test_sse_returns_last_event(self):
        first = {"jsonrpc": "2.0", "method": "notifications/progress"}
        second = {"jsonrpc": "2.0", "id": "1", "result": {"done": True}}
        sse_body = _sse_body(first) + _sse_body(second)
        response = httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )
        assert BaseAdapter._parse_response(response) == second

    def test_sse_raises_on_empty_body(self):
        response = httpx.Response(
            200,
            text="",
            headers={"content-type": "text/event-stream"},
        )
        with pytest.raises(ValueError, match="No data field found"):
            BaseAdapter._parse_response(response)


class TestInitializeSSE:
    @respx.mock
    async def test_initialize_parses_sse_response(self, base_url):
        init_result = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "test-server", "version": "1.0"},
            },
        }
        sse_body = _sse_body(init_result)
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    text=sse_body,
                    headers={
                        "content-type": "text/event-stream",
                        "Mcp-Session-Id": "sess-sse-1",
                    },
                ),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(base_url)
        result = await adapter.initialize()

        assert result == init_result
        assert adapter._session_id == "sess-sse-1"
        assert adapter._protocol_version == PROTOCOL_VERSION

    @respx.mock
    async def test_make_request_parses_sse_response(self, base_url):
        tools_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"tools": [{"name": "my_tool"}]},
        }
        sse_body = _sse_body(tools_result)
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200,
                text=sse_body,
                headers={"content-type": "text/event-stream"},
            ),
        )

        adapter = BaseAdapter(base_url)
        result = await adapter._make_request(
            {"id": "2", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        assert result == tools_result
