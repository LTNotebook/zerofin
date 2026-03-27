"""Test a single verification call. Use this to check a model works
before running the full 50-case test.

Run with: python scripts/test_single.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["LLM_PROVIDER"] = "openrouter"
os.environ["LLM_MODEL"] = "deepseek/deepseek-chat"

from zerofin.ai.verification import build_verification_chain

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

print(f"Verdict:      {result.verdict}")
print(f"Confidence:   {result.confidence}")
print(f"Mechanism:    {result.mechanism}")
print(f"Alternatives: {result.alternative_explanations}")
print(f"Category:     {result.relationship_category}")
print(f"Reasoning:    {result.reasoning}")
