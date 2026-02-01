# MCPHero - MCP as tools / MCP as functions

Library to use MCP as tools / functions in native AI libraries

## Inspiration

Everyone uses MCP now, but many still use old-school AI clients with no MCP support. These client libraries like `openai` or `google-genai` only have tool/function calls support.
This project is created to easily connect MCP servers to these libs as tools.

## Concept
Two main flows:
1) `list_tools` - call the MCP server over http to get the tool definitions, then map them to AI library tool definitions
2) `process_tool_calls` - get the AI library's tool_calls, parse them, send the requests to mcp servers, return results

## Installation

OpenAI (default) support:
```bash
pip install mcphero
```

For Google Gemini support:

```bash
pip install "mcphero[google-genai]"
```

## Quick Start

### OpenAI

```python
import asyncio
from openai import OpenAI
from mcphero import MCPToolAdapterOpenAI

async def main():
    adapter = MCPToolAdapterOpenAI("https://api.mcphero.app/mcp/your-server-id")
    client = OpenAI()

    # Get tool definitions
    tools = await adapter.get_tool_definitions()

    # Make request with tools
    messages = [{"role": "user", "content": "What's the weather in London?"}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    # Process tool calls if present
    if response.choices[0].message.tool_calls:
        tool_results = await adapter.process_tool_calls(
            response.choices[0].message.tool_calls
        )

        # Continue conversation with results
        messages.append(response.choices[0].message)
        messages.extend(tool_results)

        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        print(final_response.choices[0].message.content)

asyncio.run(main())
```

### Google Gemini

```python
import asyncio
from google import genai
from google.genai import types
from mcphero import MCPToolAdapterGemini

async def main():
    adapter = MCPToolAdapterGemini("https://api.mcphero.app/mcp/your-server-id")
    client = genai.Client(api_key="your-api-key")

    # Get tool definitions
    tool = await adapter.get_tool()

    # Make request with tools
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What's the weather in London?",
        config=types.GenerateContentConfig(
            tools=[tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    # Process function calls if present
    if response.function_calls:
        results = await adapter.process_function_calls(response.function_calls)

        # Continue conversation with results
        contents = [
            types.Content(role="user", parts=[types.Part.from_text("What's the weather in London?")]),
            response.candidates[0].content,
            *results,
        ]

        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[tool]),
        )
        print(final_response.text)

asyncio.run(main())
```

## Multiple MCP Servers

Adapters natively support connecting to multiple MCP servers at once. Use `MCPServerConfig` to configure each server, then pass them as a list.

### MCPServerConfig

```python
from mcphero import MCPServerConfig

config = MCPServerConfig(
    url="https://api.mcphero.app/mcp/your-server-id",  # required
    name="weather",              # optional, auto-derived from URL if omitted
    timeout=30.0,                # optional, default 30s
    headers={                    # optional, auth headers for the server
        "Authorization": "Bearer your-token",
    },
    init_mode="auto",            # "auto" | "on_fail" | "none"
    tool_prefix="wx",            # optional, prefix for tool names from this server
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | *required* | HTTP endpoint of the MCP server |
| `name` | `str \| None` | derived from URL | Identifier for the server (e.g. last path segment) |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
| `headers` | `dict[str, str] \| None` | `None` | Headers sent with every request (useful for auth) |
| `init_mode` | `"auto" \| "on_fail" \| "none"` | `"auto"` | When to run MCP initialization handshake |
| `tool_prefix` | `str \| None` | `None` | Prefix applied to all tool names from this server |

**`init_mode` options:**
- `"auto"` - initialize the connection before every request (default, safest)
- `"on_fail"` - skip initialization, but retry with initialization if a request fails
- `"none"` - never initialize (for servers that don't require it)

### Multi-Server Example

```python
import asyncio
from openai import OpenAI
from mcphero import MCPToolAdapterOpenAI, MCPServerConfig

async def main():
    adapter = MCPToolAdapterOpenAI([
        MCPServerConfig(
            url="https://api.mcphero.app/mcp/weather",
            name="weather",
            headers={"Authorization": "Bearer weather-token"},
        ),
        MCPServerConfig(
            url="https://api.mcphero.app/mcp/calendar",
            name="calendar",
            headers={"Authorization": "Bearer calendar-token"},
        ),
    ])

    client = OpenAI()

    # Tools from ALL servers are fetched in parallel and merged
    tools = await adapter.get_tool_definitions()

    messages = [{"role": "user", "content": "What's the weather today and what's on my calendar?"}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    # Tool calls are automatically routed to the correct server
    if response.choices[0].message.tool_calls:
        results = await adapter.process_tool_calls(
            response.choices[0].message.tool_calls
        )
        messages.append(response.choices[0].message)
        messages.extend(results)

        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        print(final_response.choices[0].message.content)

asyncio.run(main())
```

### Tool Name Collisions

When multiple servers expose tools with the same name, the adapter auto-prefixes them with the server name to avoid collisions:

```python
# Both servers have a "search" tool
adapter = MCPToolAdapterOpenAI([
    MCPServerConfig(url="https://example.com/mcp/weather", name="weather"),
    MCPServerConfig(url="https://example.com/mcp/calendar", name="calendar"),
])

tools = await adapter.get_tool_definitions()
# "search" becomes "weather__search" and "calendar__search"
```

You can control this behavior:

```python
# Custom separator
adapter = MCPToolAdapterOpenAI(configs, prefix_separator="-")
# "weather-search", "calendar-search"

# Disable auto-prefixing (will raise on collision)
adapter = MCPToolAdapterOpenAI(configs, auto_prefix_on_collision=False)

# Manual prefix via config (always applied, regardless of collisions)
MCPServerConfig(url="...", tool_prefix="wx")
# "wx__search"
```

## API Reference

### MCPToolAdapterOpenAI

```python
from mcphero import MCPToolAdapterOpenAI, MCPServerConfig

# Single server (URL string)
adapter = MCPToolAdapterOpenAI("https://api.mcphero.app/mcp/your-server-id")

# Single server (config)
adapter = MCPToolAdapterOpenAI(
    MCPServerConfig(
        url="https://api.mcphero.app/mcp/your-server-id",
        headers={"Authorization": "Bearer ..."},
    )
)

# Multiple servers
adapter = MCPToolAdapterOpenAI([
    MCPServerConfig(url="https://server-a.com/mcp", name="a"),
    MCPServerConfig(url="https://server-b.com/mcp", name="b"),
])
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_tool_definitions()` | `list[ChatCompletionToolParam]` | Fetch tools from MCP server(s) as OpenAI tool schemas |
| `process_tool_calls(tool_calls, return_errors=True)` | `list[ChatCompletionToolMessageParam]` | Execute tool calls and return results for the conversation |
| `discover_tools()` | `list[MCPToolDefinition]` | Low-level: discover tools with routing metadata |
| `call_tool(name, arguments)` | `JsonRpcResponse` | Low-level: call a single tool by name |
| `initialize_all()` | `dict[str, JsonRpcResponse \| Exception]` | Pre-initialize all server connections |

### MCPToolAdapterGemini

```python
from mcphero import MCPToolAdapterGemini, MCPServerConfig

# Same constructor options as OpenAI adapter
adapter = MCPToolAdapterGemini("https://api.mcphero.app/mcp/your-server-id")
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_function_declarations()` | `list[types.FunctionDeclaration]` | Fetch tools as Gemini FunctionDeclaration objects |
| `get_tool()` | `types.Tool` | Fetch tools as a Gemini Tool object |
| `process_function_calls(function_calls, return_errors=True)` | `list[types.Content]` | Execute function calls and return Content objects |
| `process_function_calls_as_parts(function_calls, return_errors=True)` | `list[types.Part]` | Execute function calls and return Part objects |
| `discover_tools()` | `list[MCPToolDefinition]` | Low-level: discover tools with routing metadata |
| `call_tool(name, arguments)` | `JsonRpcResponse` | Low-level: call a single tool by name |

## Error Handling

Both adapters handle errors gracefully. When `return_errors=True` (default), failed tool calls return error messages that can be sent back to the model:

```python
# Tool call fails -> returns error in result
results = await adapter.process_tool_calls(tool_calls, return_errors=True)
# [{"role": "tool", "tool_call_id": "...", "content": "{\"error\": \"HTTP error...\"}"}]

# Skip failed calls
results = await adapter.process_tool_calls(tool_calls, return_errors=False)
```

## Links

- [Website](https://mcphero.app)
- [Documentation](https://mcphero.app/docs)
- [GitHub](https://github.com/stepacool/mcphero)
- [Issues](https://github.com/stepacool/mcphero/issues)

## License

MIT

# Need a custom MCP server? Or a good, no bloat MCP server? Visit [MCPHero](https://mcphero.app) and create one!
