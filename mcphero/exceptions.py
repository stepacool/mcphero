class MCPHeroException(Exception):
    """Base MCPHero Exception class"""


INSTALL_OPENAI = """
To use OpenAI with mcphero, please install dependencies:

pip install "mcphero[openai]"
"""

INSTALL_GOOGLE_GENAI = """
To use Google Gemini with mcphero, please install dependencies:

pip install "mcphero[google-genai]"
"""
