from mcphero.adapters.base_adapter import BaseAdapter
from mcphero.adapters.generic import GenericToolCall, GenericToolResult, MCPToolAdapter

__all__ = [
    "BaseAdapter",
    "GenericToolCall",
    "GenericToolResult",
    "MCPToolAdapter",
    "MCPToolAdapterOpenAI",  # pyright: ignore[reportUnsupportedDunderAll]
    "MCPToolAdapterGemini",  # pyright: ignore[reportUnsupportedDunderAll]
]


def __getattr__(name: str):
    if name == "MCPToolAdapterOpenAI":
        from mcphero.adapters.openai import MCPToolAdapterOpenAI

        return MCPToolAdapterOpenAI
    if name == "MCPToolAdapterGemini":
        from mcphero.adapters.gemini import MCPToolAdapterGemini

        return MCPToolAdapterGemini
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
