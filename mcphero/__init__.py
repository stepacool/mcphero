from mcphero.adapters.openai import MCPToolAdapterOpenAI

__all__ = [
    "MCPToolAdapterOpenAI",
    "MCPToolAdapterGemini",  # pyright: ignore[reportUnsupportedDunderAll]
]


def __getattr__(name: str):
    if name == "MCPToolAdapterGemini":
        from mcphero.adapters.gemini import MCPToolAdapterGemini

        return MCPToolAdapterGemini
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
