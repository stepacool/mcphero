"""
MCP Tool Adapter for Google Gemini.

This adapter converts remote MCP server tools to Gemini-compatible tool definitions
and processes Gemini's function calls to HTTP requests to the MCP server.

Requires the optional `google-genai` dependency:
    pip install mcphero[google-genai]
"""

from __future__ import annotations

try:
    from google.genai import types  # pyright: ignore[reportMissingImports]
except ImportError as e:
    from mcphero.exceptions import INSTALL_GOOGLE_GENAI

    raise ImportError(INSTALL_GOOGLE_GENAI) from e

import httpx

from mcphero.adapters.base_adapter import BaseAdapter


class MCPToolAdapterGemini(BaseAdapter):
    """
    Adapter that converts remote MCP server tools to Gemini-compatible tool definitions.
    Also converts Gemini's function calls to HTTP requests to the MCP server.

    Requires the optional `google-genai` dependency:
        pip install mcphero[google-genai]

    Usage with google-genai SDK:

        from google import genai
        from google.genai import types

        adapter = MCPToolAdapterGemini("https://api.mcphero.app/mcp/your-server-id")

        # Get tool definitions
        tool = await adapter.get_tool()

        # Create Gemini client and make request
        client = genai.Client(api_key="...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What's the weather?",
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
            # Add results to conversation and continue
    """

    async def get_function_declarations(self) -> list[types.FunctionDeclaration]:
        """
        Fetch tools from MCP server and convert them to Gemini FunctionDeclaration objects.

        Returns:
            List of FunctionDeclaration objects.
        """
        mcp_tools = await self.get_mcp_tools()

        declarations: list[types.FunctionDeclaration] = []
        for tool in mcp_tools:
            declaration = types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get(
                    "inputSchema",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
            declarations.append(declaration)

        return declarations

    async def get_tool(self) -> types.Tool:
        """
        Fetch tools from MCP server and return as a Gemini Tool object.

        This returns a Tool that can be passed directly to GenerateContentConfig.

        Returns:
            A Tool object containing all function declarations.
        """
        declarations = await self.get_function_declarations()
        return types.Tool(function_declarations=declarations)

    async def process_function_calls(
        self,
        function_calls: list[types.FunctionCall],
        return_errors: bool = True,
    ) -> list[types.Content]:
        """
        Process Gemini's function calls to invoke the tools via HTTP.

        Args:
            function_calls: List of FunctionCall objects from the Gemini response.
            return_errors: If True, include error responses for failed calls.

        Returns:
            List of Content objects containing function responses that can be
            added to the conversation.

        Example:
            if response.function_calls:
                results = await adapter.process_function_calls(response.function_calls)

                # Add to conversation:
                contents.append(response.candidates[0].content)  # Model's function call
                contents.extend(results)  # Function responses
        """
        results: list[types.Content] = []

        for fc in function_calls:
            tool_name = fc.name
            arguments = fc.args if fc.args else {}
            call_id = getattr(fc, "id", None)

            try:
                result = await self.call_mcp_tool(tool_name, arguments)

                function_response = types.FunctionResponse(
                    name=tool_name,
                    response={"result": result}
                    if not isinstance(result, dict)
                    else result,
                    id=call_id,
                )

                content = types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=function_response.name,
                            response=function_response.response,
                        )
                    ],
                )
                results.append(content)

            except httpx.HTTPError as e:
                if return_errors:
                    function_response = types.FunctionResponse(
                        name=tool_name,
                        response={"error": f"HTTP error calling tool: {str(e)}"},
                        id=call_id,
                    )
                    content = types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=function_response.name,
                                response=function_response.response,
                            )
                        ],
                    )
                    results.append(content)

            except Exception as e:
                if return_errors:
                    function_response = types.FunctionResponse(
                        name=tool_name,
                        response={"error": f"Unexpected error: {str(e)}"},
                        id=call_id,
                    )
                    content = types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=function_response.name,
                                response=function_response.response,
                            )
                        ],
                    )
                    results.append(content)

        return results

    async def process_function_calls_as_parts(
        self,
        function_calls: list[types.FunctionCall],
        return_errors: bool = True,
    ) -> list[types.Part]:
        """
        Process Gemini's function calls and return just the Part objects.

        This is useful when you want to construct the Content object yourself
        or combine multiple function responses into a single Content.

        Args:
            function_calls: List of FunctionCall objects from the Gemini response.
            return_errors: If True, include error responses for failed calls.

        Returns:
            List of Part objects containing function responses.

        Example:
            parts = await adapter.process_function_calls_as_parts(function_calls)

            # Combine into single Content:
            contents.append(types.Content(role="user", parts=parts))
        """
        results: list[types.Part] = []

        for fc in function_calls:
            tool_name = fc.name
            arguments = fc.args if fc.args else {}

            try:
                result = await self.call_mcp_tool(tool_name, arguments)

                part = types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                    if not isinstance(result, dict)
                    else result,
                )
                results.append(part)

            except httpx.HTTPError as e:
                if return_errors:
                    part = types.Part.from_function_response(
                        name=tool_name,
                        response={"error": f"HTTP error calling tool: {str(e)}"},
                    )
                    results.append(part)

            except Exception as e:
                if return_errors:
                    part = types.Part.from_function_response(
                        name=tool_name,
                        response={"error": f"Unexpected error: {str(e)}"},
                    )
                    results.append(part)

        return results

    @staticmethod
    def create_function_response_content(
        name: str,
        response: dict,
    ) -> types.Content:
        """
        Helper to create a function response Content object.

        Args:
            name: The function name that was called.
            response: The response data from the function.

        Returns:
            A Content object that can be appended to the conversation.
        """
        return types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=name, response=response)],
        )
