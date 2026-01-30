import pytest


@pytest.fixture
def sample_mcp_tools():
    """Sample MCP tool definitions as returned by a server."""
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
        {
            "name": "search",
            "description": "Search the web",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        },
    ]


@pytest.fixture
def sample_mcp_tool_no_schema():
    """MCP tool without inputSchema."""
    return [
        {
            "name": "ping",
            "description": "Ping the server",
        },
    ]


@pytest.fixture
def sample_tool_result():
    """Sample successful tool call result."""
    return {"temperature": 72, "unit": "fahrenheit", "description": "Sunny"}


@pytest.fixture
def base_url():
    return "https://api.mcphero.app/mcp/test-server"
