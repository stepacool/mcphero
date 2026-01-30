import httpx
import pytest
import respx

from mcphero.adapters.base_adapter import BaseAdapter


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


class TestMakeRequest:
    @respx.mock
    async def test_get_returns_parsed_json(self, base_url):
        expected = [{"name": "tool1"}]
        respx.get(f"{base_url}/tools").mock(
            return_value=httpx.Response(200, json=expected)
        )

        adapter = BaseAdapter(base_url)
        result = await adapter.make_request("/tools")
        assert result == expected

    @respx.mock
    async def test_post_sends_json_body(self, base_url):
        expected = {"result": "ok"}
        route = respx.post(f"{base_url}/tools/test/call").mock(
            return_value=httpx.Response(200, json=expected)
        )

        adapter = BaseAdapter(base_url)
        result = await adapter.make_request(
            "/tools/test/call", method="POST", data={"arguments": {"q": "hello"}}
        )
        assert result == expected
        assert route.called

    @respx.mock
    async def test_unsupported_method_raises_value_error(self, base_url):
        adapter = BaseAdapter(base_url)
        with pytest.raises(ValueError, match="Unsupported HTTP method: DELETE"):
            await adapter.make_request("/tools", method="DELETE")

    @respx.mock
    async def test_http_error_raises(self, base_url):
        respx.get(f"{base_url}/tools").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        adapter = BaseAdapter(base_url)
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.make_request("/tools")

    @respx.mock
    async def test_get_passes_params(self, base_url):
        route = respx.get(f"{base_url}/search", params={"q": "test"}).mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        adapter = BaseAdapter(base_url)
        result = await adapter.make_request("/search", params={"q": "test"})
        assert result == {"results": []}
        assert route.called
