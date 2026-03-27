"""Test a single verification call. Use this to check a model works
before running the full 50-case test.

Run with: python scripts/tests/test_single.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

os.environ["LLM_PROVIDER"] = "openrouter"
os.environ["LLM_MODEL"] = "deepseek/deepseek-chat"

from zerofin.ai.verification import build_verification_chain  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

chain = build_verification_chain()
result = chain.invoke({
    "entity_a_id": "NVDA",
    "entity_a_desc": "NVIDIA — semiconductor company, GPUs, AI accelerators",
    "entity_a_type": "asset",
    "entity_b_id": "AMD",
    "entity_b_desc": "Advanced Micro Devices — semiconductor company, GPUs, CPUs",
    "entity_b_type": "asset",
    "correlation": 0.12,
    "direction": "positive",
    "window_days": 252,
    "observation_count": 189,
})

logger.info("Verdict:      %s", result.verdict)
logger.info("Confidence:   %s", result.confidence)
logger.info("Mechanism:    %s", result.mechanism)
logger.info("Alternatives: %s", result.alternative_explanations)
logger.info("Category:     %s", result.relationship_category)
logger.info("Reasoning:    %s", result.reasoning)
