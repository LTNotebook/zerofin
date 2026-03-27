"""Test the two-pass verification chain (DeepSeek + Sonnet review).

Sends a few pairs through pass 1, then routes borderline ones to pass 2.
Uses real API calls — costs a few cents.

Run with: python scripts/tests/test_two_pass.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from zerofin.ai.verification import (  # noqa: E402
    build_review_chain,
    build_verification_chain,
    needs_second_pass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pairs designed to trigger different outcomes:
# - One obvious (should skip pass 2)
# - One that DeepSeek typically gets wrong (CEG/NVDA — knowledge cutoff)
# - One borderline (KMI/O — income/yield pair)
# - One nonsense (should skip pass 2)
TEST_PAIRS = [
    {
        "label": "NVDA vs AMD (obvious — should skip pass 2)",
        "entity_a_id": "NVDA",
        "entity_a_desc": "NVIDIA — GPUs, AI accelerators, data center chips",
        "entity_a_type": "asset",
        "entity_b_id": "AMD",
        "entity_b_desc": "Advanced Micro Devices — GPUs, CPUs, data center chips",
        "entity_b_type": "asset",
        "correlation": 0.12,
        "direction": "positive",
        "window_days": 252,
        "observation_count": 189,
    },
    {
        "label": "CEG vs NVDA (knowledge cutoff — DeepSeek misses AI power)",
        "entity_a_id": "CEG",
        "entity_a_desc": "Constellation Energy — nuclear and natural gas power generation",
        "entity_a_type": "asset",
        "entity_b_id": "NVDA",
        "entity_b_desc": "NVIDIA — GPUs, AI accelerators, data center chips",
        "entity_b_type": "asset",
        "correlation": 0.13,
        "direction": "positive",
        "window_days": 252,
        "observation_count": 189,
    },
    {
        "label": "KMI vs O (income/yield — borderline)",
        "entity_a_id": "KMI",
        "entity_a_desc": "Kinder Morgan — natural gas pipeline and storage infrastructure",
        "entity_a_type": "asset",
        "entity_b_id": "O",
        "entity_b_desc": "Realty Income — net lease REIT, bond-like monthly dividends",
        "entity_b_type": "asset",
        "correlation": 0.11,
        "direction": "positive",
        "window_days": 252,
        "observation_count": 189,
    },
    {
        "label": "NFLX vs ZW=F (nonsense — should skip pass 2)",
        "entity_a_id": "NFLX",
        "entity_a_desc": "Netflix — streaming entertainment",
        "entity_a_type": "asset",
        "entity_b_id": "ZW=F",
        "entity_b_desc": "Wheat Futures (CBOT)",
        "entity_b_type": "asset",
        "correlation": 0.10,
        "direction": "positive",
        "window_days": 63,
        "observation_count": 47,
    },
]


def main() -> None:
    chain = build_verification_chain()

    logger.info("=== PASS 1: DeepSeek ===\n")
    pass1_results = []
    for pair in TEST_PAIRS:
        label = pair["label"]
        inputs = {k: v for k, v in pair.items() if k != "label"}
        result = chain.invoke(inputs)
        pass1_results.append((pair, result))

        review = "-> ROUTE TO PASS 2" if needs_second_pass(result) else "-> DONE"
        logger.info("%s", label)
        logger.info("  Verdict: %s (%.2f) %s", result.verdict, result.confidence, review)
        logger.info("  Mechanism: %s", result.mechanism[:100])
        logger.info("")

    # Collect borderline cases for pass 2
    borderline = [(pair, result) for pair, result in pass1_results if needs_second_pass(result)]

    if not borderline:
        logger.info("No borderline cases — pass 2 skipped.")
        return

    logger.info("=== PASS 2: Sonnet (reviewing %d borderline cases) ===\n", len(borderline))
    review_chain = build_review_chain()

    for pair, pass1_result in borderline:
        label = pair["label"]
        review_input = {
            "entity_a_id": pair["entity_a_id"],
            "entity_a_desc": pair["entity_a_desc"],
            "entity_a_type": pair["entity_a_type"],
            "entity_b_id": pair["entity_b_id"],
            "entity_b_desc": pair["entity_b_desc"],
            "entity_b_type": pair["entity_b_type"],
            "correlation": pair["correlation"],
            "direction": pair["direction"],
            "window_days": pair["window_days"],
            "observation_count": pair["observation_count"],
            "pass1_verdict": pass1_result.verdict,
            "pass1_confidence": pass1_result.confidence,
            "pass1_mechanism": pass1_result.mechanism,
            "pass1_reasoning": pass1_result.reasoning,
        }

        review_result = review_chain.invoke(review_input)
        changed = "OVERRIDE" if review_result.verdict != pass1_result.verdict else "AGREED"

        logger.info("%s", label)
        logger.info("  Pass 1: %s (%.2f)", pass1_result.verdict, pass1_result.confidence)
        logger.info("  Pass 2: %s (%.2f) [%s]", review_result.verdict, review_result.confidence, changed)
        logger.info("  Mechanism: %s", review_result.mechanism[:150])
        logger.info("")


if __name__ == "__main__":
    main()
