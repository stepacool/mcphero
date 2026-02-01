import json

import httpx
import pytest
import respx

from mcphero.__about__ import __version__
from mcphero.adapters.base_adapter import (
    PROTOCOL_VERSION,
    BaseAdapter,
    InitMode,
    MCPConnection,
    MCPServerConfig,
)


def _sse_body(payload: dict) -> str:
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


def _init_response(version=PROTOCOL_VERSION):
    return {"jsonrpc": "2.0", "id": "1", "result": {"protocolVersion": version}}


def _tools_response(tools):
    return {"jsonrpc": "2.0", "id": "1", "result": {"tools": tools}}


# ── MCPServerConfig ──────────────────────────────────────────────


class TestMCPServerConfig:
    def test_auto_name_from_url(self):
        cfg = MCPServerConfig(url="https://example.com/mcp/weather")
        assert cfg.name == "weather"

    def test_auto_name_strips_trailing_slash(self):
        cfg = MCPServerConfig(url="https://example.com/mcp/weather/")
        assert cfg.name == "weather"

    def test_explicit_name(self):
        cfg = MCPServerConfig(url="https://example.com/x", name="my-srv")
        assert cfg.name == "my-srv"

    def test_default_timeout(self):
        cfg = MCPServerConfig(url="https://example.com/x")
        assert cfg.timeout == 30.0

    def test_custom_timeout(self):
        cfg = MCPServerConfig(url="https://example.com/x", timeout=60.0)
        assert cfg.timeout == 60.0

    def test_default_headers(self):
        cfg = MCPServerConfig(url="https://example.com/x")
        assert cfg.headers is None

    def test_custom_headers(self):
        h = {"Authorization": "Bearer token"}
        cfg = MCPServerConfig(url="https://example.com/x", headers=h)
        assert cfg.headers == h

    def test_default_init_mode(self):
        cfg = MCPServerConfig(url="https://example.com/x")
        assert cfg.init_mode == InitMode.auto

    def test_string_init_mode(self):
        cfg = MCPServerConfig(url="https://example.com/x", init_mode="on_fail")
        assert cfg.init_mode == InitMode.on_fail

    def test_enum_init_mode(self):
        cfg = MCPServerConfig(url="https://example.com/x", init_mode=InitMode.none)
        assert cfg.init_mode == InitMode.none

    def test_default_tool_prefix(self):
        cfg = MCPServerConfig(url="https://example.com/x")
        assert cfg.tool_prefix is None

    def test_custom_tool_prefix(self):
        cfg = MCPServerConfig(url="https://example.com/x", tool_prefix="wx")
        assert cfg.tool_prefix == "wx"


# ── MCPConnection: initialize ────────────────────────────────────


class TestMCPConnectionInitialize:
    @respx.mock
    async def test_sends_correct_payload(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        result = await conn.initialize()

        assert result == _init_response()
        assert route.call_count == 2

        payload = json.loads(route.calls[0].request.content)
        assert payload["method"] == "initialize"
        assert payload["params"]["protocolVersion"] == PROTOCOL_VERSION
        assert payload["params"]["clientInfo"]["name"] == "mcphero"
        assert payload["params"]["clientInfo"]["version"] == __version__

    @respx.mock
    async def test_extracts_session_id(self, base_url):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "sess-42"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        assert conn._session_id == "sess-42"

    @respx.mock
    async def test_sends_initialized_notification(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        notification = json.loads(route.calls[1].request.content)
        assert notification["method"] == "notifications/initialized"
        assert "id" not in notification

    @respx.mock
    async def test_notification_includes_session_id(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "abc"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        assert route.calls[1].request.headers.get("Mcp-Session-Id") == "abc"

    @respx.mock
    async def test_idempotent(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        r1 = await conn.initialize()
        r2 = await conn.initialize()

        assert r1 == r2
        assert route.call_count == 2

    @respx.mock
    async def test_works_without_session_id(self, base_url):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_init_response()),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        assert conn._session_id is None

    @respx.mock
    async def test_stores_negotiated_protocol_version(self, base_url):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_init_response("2024-11-05")),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        assert conn._protocol_version == "2024-11-05"

    @respx.mock
    async def test_parses_sse_response(self, base_url):
        init_result = _init_response()
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    text=_sse_body(init_result),
                    headers={
                        "content-type": "text/event-stream",
                        "Mcp-Session-Id": "sse-1",
                    },
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        result = await conn.initialize()

        assert result == init_result
        assert conn._session_id == "sse-1"


# ── MCPConnection: _make_request ─────────────────────────────────


class TestMCPConnectionMakeRequest:
    @respx.mock
    async def test_returns_parsed_json(self, base_url):
        expected = {"jsonrpc": "2.0", "id": "1", "result": {"tools": []}}
        respx.post(base_url).mock(
            return_value=httpx.Response(200, json=expected)
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="none"))
        result = await conn._make_request(
            {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        assert result == expected

    @respx.mock
    async def test_http_error_raises(self, base_url):
        respx.post(base_url).mock(
            return_value=httpx.Response(500, text="Error")
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="none"))
        with pytest.raises(httpx.HTTPStatusError):
            await conn._make_request(
                {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
            )

    @respx.mock
    async def test_no_session_header_before_init(self, base_url):
        route = respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json={"jsonrpc": "2.0", "id": "1", "result": {}}
            )
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="none"))
        await conn._make_request(
            {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        assert "Mcp-Session-Id" not in route.calls[0].request.headers

    @respx.mock
    async def test_includes_session_after_init(self, base_url):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "sess-xyz"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        respx.reset()
        route = respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json={"jsonrpc": "2.0", "id": "2", "result": {}}
            )
        )

        await conn._make_request(
            {"id": "2", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        assert route.calls[0].request.headers.get("Mcp-Session-Id") == "sess-xyz"

    @respx.mock
    async def test_includes_protocol_version_after_init(self, base_url):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_init_response("2024-11-05")),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url))
        await conn.initialize()

        respx.reset()
        route = respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json={"jsonrpc": "2.0", "id": "2", "result": {}}
            )
        )

        await conn._make_request(
            {"id": "2", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        assert (
            route.calls[0].request.headers.get("MCP-Protocol-Version") == "2024-11-05"
        )

    @respx.mock
    async def test_parses_sse_response(self, base_url):
        result = {"jsonrpc": "2.0", "id": "1", "result": {"tools": [{"name": "t"}]}}
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(result),
                headers={"content-type": "text/event-stream"},
            )
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="none"))
        r = await conn._make_request(
            {"id": "1", "jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )

        assert r == result


# ── Init modes (on MCPConnection) ────────────────────────────────


class TestInitModeAuto:
    @respx.mock
    async def test_initializes_before_get_tools(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
                httpx.Response(200, json=_tools_response([])),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="auto"))
        tools = await conn.get_tools()

        assert tools == []
        assert route.call_count == 3

    @respx.mock
    async def test_initializes_before_call_tool(self, base_url):
        call_result = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {"content": [{"type": "text", "text": "hi"}]},
        }
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
                httpx.Response(200, json=call_result),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="auto"))
        result = await conn.call_tool("test_tool", {"arg": "val"})

        assert result == call_result
        assert route.call_count == 3


class TestInitModeOnFail:
    @respx.mock
    async def test_skips_init_on_success(self, base_url):
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json=_tools_response([]))
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="on_fail"))
        tools = await conn.get_tools()

        assert tools == []
        assert route.call_count == 1

    @respx.mock
    async def test_initializes_and_retries_on_failure(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(400, text="Bad Request"),
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
                httpx.Response(200, json=_tools_response([])),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="on_fail"))
        tools = await conn.get_tools()

        assert tools == []
        assert route.call_count == 4

    @respx.mock
    async def test_no_retry_after_init(self, base_url):
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="on_fail"))
        await conn.initialize()

        respx.reset()
        respx.post(base_url).mock(
            return_value=httpx.Response(400, text="Bad Request")
        )

        with pytest.raises(httpx.HTTPStatusError):
            await conn.get_tools()

    @respx.mock
    async def test_error_propagates_on_retry_failure(self, base_url):
        route = respx.post(base_url).mock(
            side_effect=[
                httpx.Response(400, text="Bad Request"),
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "s1"},
                ),
                httpx.Response(202),
                httpx.Response(500, text="Internal Server Error"),
            ]
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="on_fail"))

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await conn.get_tools()

        assert exc_info.value.response.status_code == 500
        assert route.call_count == 4


class TestInitModeNone:
    @respx.mock
    async def test_no_init(self, base_url):
        route = respx.post(base_url).mock(
            return_value=httpx.Response(200, json=_tools_response([]))
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="none"))
        tools = await conn.get_tools()

        assert tools == []
        assert route.call_count == 1

    @respx.mock
    async def test_error_propagates_directly(self, base_url):
        route = respx.post(base_url).mock(
            return_value=httpx.Response(400, text="Bad Request")
        )

        conn = MCPConnection(MCPServerConfig(url=base_url, init_mode="none"))

        with pytest.raises(httpx.HTTPStatusError):
            await conn.get_tools()

        assert route.call_count == 1


# ── SSE parsing ──────────────────────────────────────────────────


class TestParseSSE:
    def test_parses_json_content_type(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {}}
        response = httpx.Response(
            200, json=payload, headers={"content-type": "application/json"}
        )
        assert MCPConnection._parse_response(response) == payload

    def test_parses_sse_content_type(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {"tools": []}}
        response = httpx.Response(
            200,
            text=_sse_body(payload),
            headers={"content-type": "text/event-stream"},
        )
        assert MCPConnection._parse_response(response) == payload

    def test_parses_sse_without_trailing_blank(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {}}
        response = httpx.Response(
            200,
            text=f"event: message\ndata: {json.dumps(payload)}",
            headers={"content-type": "text/event-stream"},
        )
        assert MCPConnection._parse_response(response) == payload

    def test_parses_sse_with_charset(self):
        payload = {"jsonrpc": "2.0", "id": "1", "result": {}}
        response = httpx.Response(
            200,
            text=_sse_body(payload),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
        assert MCPConnection._parse_response(response) == payload

    def test_returns_last_sse_event(self):
        first = {"jsonrpc": "2.0", "method": "notifications/progress"}
        second = {"jsonrpc": "2.0", "id": "1", "result": {"done": True}}
        response = httpx.Response(
            200,
            text=_sse_body(first) + _sse_body(second),
            headers={"content-type": "text/event-stream"},
        )
        assert MCPConnection._parse_response(response) == second

    def test_raises_on_empty_body(self):
        response = httpx.Response(
            200, text="", headers={"content-type": "text/event-stream"}
        )
        with pytest.raises(ValueError, match="No data field found"):
            MCPConnection._parse_response(response)


# ── BaseAdapter: construction ────────────────────────────────────


class TestBaseAdapterConstruction:
    def test_single_url_string(self, base_url):
        adapter = BaseAdapter(base_url)
        assert len(adapter._configs) == 1
        assert adapter._configs[0].url == base_url

    def test_single_config(self, base_url):
        cfg = MCPServerConfig(url=base_url, name="my-server")
        adapter = BaseAdapter(cfg)
        assert len(adapter._configs) == 1
        assert adapter._configs[0].name == "my-server"

    def test_multiple_configs(self):
        adapter = BaseAdapter(
            [
                MCPServerConfig(url="https://a.com/weather", name="weather"),
                MCPServerConfig(url="https://b.com/calendar", name="calendar"),
            ]
        )
        assert len(adapter._configs) == 2
        assert adapter.is_multi_server

    def test_single_is_not_multi_server(self, base_url):
        adapter = BaseAdapter(base_url)
        assert not adapter.is_multi_server

    def test_connections_created_per_server(self):
        adapter = BaseAdapter(
            [
                MCPServerConfig(url="https://a.com/weather", name="weather"),
                MCPServerConfig(url="https://b.com/calendar", name="calendar"),
            ]
        )
        assert "weather" in adapter._connections
        assert "calendar" in adapter._connections

    def test_tools_raises_before_discovery(self, base_url):
        adapter = BaseAdapter(base_url)
        with pytest.raises(RuntimeError, match="Must call discover_tools"):
            _ = adapter.tools


# ── BaseAdapter: discover_tools ──────────────────────────────────


class TestDiscoverTools:
    @respx.mock
    async def test_single_server(self, base_url, sample_mcp_tools):
        respx.post(base_url).mock(
            return_value=httpx.Response(200, json=_tools_response(sample_mcp_tools))
        )

        adapter = BaseAdapter(MCPServerConfig(url=base_url, init_mode="none"))
        tools = await adapter.discover_tools()

        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[0].original_name == "get_weather"
        assert tools[1].name == "search"

    @respx.mock
    async def test_multi_server_unique_tools(self):
        url_a = "https://a.com/mcp/weather"
        url_b = "https://b.com/mcp/calendar"

        respx.post(url_a).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "get_weather", "description": "Get weather"}]
                ),
            )
        )
        respx.post(url_b).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "get_events", "description": "Get events"}]
                ),
            )
        )

        adapter = BaseAdapter(
            [
                MCPServerConfig(url=url_a, name="weather", init_mode="none"),
                MCPServerConfig(url=url_b, name="calendar", init_mode="none"),
            ]
        )
        tools = await adapter.discover_tools()

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"get_weather", "get_events"}

    @respx.mock
    async def test_multi_server_collision_auto_prefix(self):
        url_a = "https://a.com/mcp/weather"
        url_b = "https://b.com/mcp/calendar"

        respx.post(url_a).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "search", "description": "Search weather"}]
                ),
            )
        )
        respx.post(url_b).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "search", "description": "Search events"}]
                ),
            )
        )

        adapter = BaseAdapter(
            [
                MCPServerConfig(url=url_a, name="weather", init_mode="none"),
                MCPServerConfig(url=url_b, name="calendar", init_mode="none"),
            ]
        )
        tools = await adapter.discover_tools()

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"weather__search", "calendar__search"}

    @respx.mock
    async def test_multi_server_collision_disabled(self):
        url_a = "https://a.com/mcp/weather"
        url_b = "https://b.com/mcp/calendar"

        respx.post(url_a).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "search", "description": "Search weather"}]
                ),
            )
        )
        respx.post(url_b).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "search", "description": "Search events"}]
                ),
            )
        )

        adapter = BaseAdapter(
            [
                MCPServerConfig(url=url_a, name="weather", init_mode="none"),
                MCPServerConfig(url=url_b, name="calendar", init_mode="none"),
            ],
            auto_prefix_on_collision=False,
        )
        tools = await adapter.discover_tools()

        names = [t.name for t in tools]
        assert names.count("search") == 2

    @respx.mock
    async def test_tool_prefix(self):
        url = "https://a.com/mcp/weather"
        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response([{"name": "search", "description": "Search"}]),
            )
        )

        adapter = BaseAdapter(
            MCPServerConfig(
                url=url, name="weather", init_mode="none", tool_prefix="wx"
            )
        )
        tools = await adapter.discover_tools()

        assert tools[0].name == "wx__search"
        assert tools[0].original_name == "search"

    @respx.mock
    async def test_custom_prefix_separator(self):
        url = "https://a.com/mcp/weather"
        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response([{"name": "search", "description": "Search"}]),
            )
        )

        adapter = BaseAdapter(
            MCPServerConfig(
                url=url, name="weather", init_mode="none", tool_prefix="wx"
            ),
            prefix_separator=".",
        )
        tools = await adapter.discover_tools()

        assert tools[0].name == "wx.search"


# ── BaseAdapter: call_tool ───────────────────────────────────────


class TestCallTool:
    @respx.mock
    async def test_routes_to_correct_server(self):
        url_a = "https://a.com/mcp/weather"
        url_b = "https://b.com/mcp/calendar"

        weather_route = respx.post(url_a).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_tools_response(
                        [{"name": "get_weather", "description": "Weather"}]
                    ),
                ),
                httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": "2", "result": {"temp": 72}}
                ),
            ]
        )
        calendar_route = respx.post(url_b).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "get_events", "description": "Events"}]
                ),
            )
        )

        adapter = BaseAdapter(
            [
                MCPServerConfig(url=url_a, name="weather", init_mode="none"),
                MCPServerConfig(url=url_b, name="calendar", init_mode="none"),
            ]
        )
        await adapter.discover_tools()

        result = await adapter.call_tool("get_weather", {"location": "NYC"})

        assert result["result"]["temp"] == 72
        assert weather_route.call_count == 2  # discovery + call
        assert calendar_route.call_count == 1  # discovery only

    @respx.mock
    async def test_unknown_tool_raises(self, base_url, sample_mcp_tools):
        respx.post(base_url).mock(
            return_value=httpx.Response(
                200, json=_tools_response(sample_mcp_tools)
            )
        )

        adapter = BaseAdapter(MCPServerConfig(url=base_url, init_mode="none"))
        await adapter.discover_tools()

        with pytest.raises(KeyError, match="Unknown tool"):
            await adapter.call_tool("nonexistent", {})

    @respx.mock
    async def test_auto_discovers_if_needed(self, base_url, sample_mcp_tools):
        call_response = {"jsonrpc": "2.0", "id": "2", "result": {"temp": 72}}
        respx.post(base_url).mock(
            side_effect=[
                httpx.Response(200, json=_tools_response(sample_mcp_tools)),
                httpx.Response(200, json=call_response),
            ]
        )

        adapter = BaseAdapter(MCPServerConfig(url=base_url, init_mode="none"))
        result = await adapter.call_tool("get_weather", {"location": "NYC"})

        assert result == call_response

    @respx.mock
    async def test_routes_prefixed_tool_to_original_name(self):
        url_a = "https://a.com/mcp/weather"
        url_b = "https://b.com/mcp/calendar"

        respx.post(url_a).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_tools_response(
                        [{"name": "search", "description": "Search weather"}]
                    ),
                ),
                httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": "2", "result": {"data": "wx"}},
                ),
            ]
        )
        respx.post(url_b).mock(
            return_value=httpx.Response(
                200,
                json=_tools_response(
                    [{"name": "search", "description": "Search events"}]
                ),
            )
        )

        adapter = BaseAdapter(
            [
                MCPServerConfig(url=url_a, name="weather", init_mode="none"),
                MCPServerConfig(url=url_b, name="calendar", init_mode="none"),
            ]
        )
        await adapter.discover_tools()

        # Call using the prefixed name
        result = await adapter.call_tool("weather__search", {"q": "rain"})
        assert result["result"]["data"] == "wx"


# ── BaseAdapter: initialize_all ──────────────────────────────────


class TestInitializeAll:
    @respx.mock
    async def test_initializes_all_connections(self):
        url_a = "https://a.com/mcp/weather"
        url_b = "https://b.com/mcp/calendar"

        respx.post(url_a).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "sa"},
                ),
                httpx.Response(202),
            ]
        )
        respx.post(url_b).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_init_response(),
                    headers={"Mcp-Session-Id": "sb"},
                ),
                httpx.Response(202),
            ]
        )

        adapter = BaseAdapter(
            [
                MCPServerConfig(url=url_a, name="weather"),
                MCPServerConfig(url=url_b, name="calendar"),
            ]
        )
        results = await adapter.initialize_all()

        assert "weather" in results
        assert "calendar" in results
        assert not isinstance(results["weather"], Exception)
        assert not isinstance(results["calendar"], Exception)
