# MCPHero - MCP as tools / MCP as functions

Library to use MCP as tools / functions in native AI libraries

## Inspiration

Everyone uses MCP now, but many still use old-school AI clients with no MCP support. These client libraries like `openai` or `google-genai` only have tool/function calls support.
This project is created to easily connect MCP servers to these libs as tools.

## Concept
Two main flows:
1) `list_tools` - call the MCP server over http to get the tool definitions, then map them to AI library tool definitions
2) `process_tool_calls' - get the AI library's tool_calls, parse them, send the requests to mcp servers, return results

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
from mcphero.adapters.openai import MCPToolAdapterOpenAI

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
from mcphero.adapters.gemini import MCPToolAdapterGemini

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

## API Reference

### MCPToolAdapterOpenAI

```python
from mcphero.adapters.openai import MCPToolAdapterOpenAI

adapter = MCPToolAdapterOpenAI(
    base_url="https://api.mcphero.app/mcp/your-server-id",
    timeout=30.0,  # optional
    headers={"Authorization": "Bearer ..."},  # optional
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_tool_definitions()` | `list[ChatCompletionToolParam]` | Fetch tools from MCP server as OpenAI tool schemas |
| `process_tool_calls(tool_calls, return_errors=True)` | `list[ChatCompletionToolMessageParam]` | Execute tool calls and return results for the conversation |

### MCPToolAdapterGemini

```python
from mcphero.adapters.gemini import MCPToolAdapterGemini

adapter = MCPToolAdapterGemini(
    base_url="https://api.mcphero.app/mcp/your-server-id",
    timeout=30.0,  # optional
    headers={"Authorization": "Bearer ..."},  # optional
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_function_declarations()` | `list[types.FunctionDeclaration]` | Fetch tools as Gemini FunctionDeclaration objects |
| `get_tool()` | `types.Tool` | Fetch tools as a Gemini Tool object |
| `process_function_calls(function_calls, return_errors=True)` | `list[types.Content]` | Execute function calls and return Content objects |
| `process_function_calls_as_parts(function_calls, return_errors=True)` | `list[types.Part]` | Execute function calls and return Part objects |

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
