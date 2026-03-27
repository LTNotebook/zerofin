"""Quick smoke test — verify LLM API keys are working."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path so we can import zerofin
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from zerofin.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_PROMPT = "What sector does NVDA (Nvidia) belong to? Reply in one sentence."


def test_openrouter() -> None:
    """Test OpenRouter API key with DeepSeek free tier."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="deepseek/deepseek-chat",
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
    )
    response = llm.invoke(TEST_PROMPT)
    logger.info("[OpenRouter/DeepSeek] %s", response.content)


def test_groq() -> None:
    """Test Groq API key with Llama 4 Scout."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.GROQ_API_KEY,
    )
    response = llm.invoke(TEST_PROMPT)
    logger.info("[Groq/Llama4] %s", response.content)


def test_voyage() -> None:
    """Test Voyage AI API key via raw HTTP (SDK has Pydantic 3.14 bug)."""
    import httpx

    response = httpx.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.VOYAGE_API_KEY}"},
        json={
            "input": ["Nvidia reported strong data center revenue growth."],
            "model": "voyage-finance-2",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    dims = len(data["data"][0]["embedding"])
    logger.info("[Voyage] Embedding returned %d dimensions", dims)


if __name__ == "__main__":
    providers = {
        "openrouter": test_openrouter,
        "groq": test_groq,
        "voyage": test_voyage,
    }

    for name, test_fn in providers.items():
        try:
            test_fn()
            logger.info("  PASS: %s", name)
        except Exception as e:
            logger.error("  FAIL: %s -- %s", name, e)
