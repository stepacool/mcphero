from __future__ import annotations

import httpx


class BaseAdapter:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict | list:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self.headers,
        ) as client:
            if method.upper() == "GET":
                response = await client.get(endpoint, params=params)
            elif method.upper() == "POST":
                response = await client.post(endpoint, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

    async def get_mcp_tools(self) -> list[dict] | dict:
        return await self._make_request(
            "",
            "POST",
            data={
                "method": "tools/list",
                "params": {},
            },
        )

    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> list | dict:
        return await self._make_request(
            "",
            "POST",
            data={
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            },
        )
