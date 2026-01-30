import abc


class BaseAdapter(abc.ABC):

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

    async def make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict:
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
