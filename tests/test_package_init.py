import pytest


class TestPackageInit:
    def test_openai_adapter_importable(self):
        from mcphero import MCPToolAdapterOpenAI

        assert MCPToolAdapterOpenAI is not None

    def test_all_contains_both_adapters(self):
        import mcphero

        assert "MCPToolAdapterOpenAI" in mcphero.__all__
        assert "MCPToolAdapterGemini" in mcphero.__all__

    def test_invalid_attribute_raises(self):
        import mcphero

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = mcphero.NoSuchAdapter
