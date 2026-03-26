"""
Base class for all data collectors.

This is the template that every data source follows. Whether it's
stock prices from yfinance, economic data from FRED, or news from
RSS feeds — they all follow this same pattern:

1. Go get data from the internet
2. Validate it through Pydantic
3. Store it in the database
4. Return a summary of what happened

By having a common template, the daily pipeline can treat all
collectors the same way without caring about the details:

    for collector in all_collectors:
        result = collector.collect_latest()
        logger.info(result["stored"])  # same format every time

New collectors just fill in the blanks — the template handles
the structure.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class CollectorResult(TypedDict, total=False):
    """The standard return shape every collector must produce.

    Using TypedDict gives us type-checked keys while still being a plain
    dict at runtime — no extra dependencies, no overhead.

    `total=False` means all keys are optional at the type level, which
    lets subclasses add extra keys (like `details` or `skipped_null`)
    without breaking the type checker. The three core keys below are
    always present in practice because _build_summary() always sets them.

    Keys:
        collector: The name of the collector (e.g. "prices", "economic").
        stored:    Number of data points successfully saved to the database.
        failed:    Number of data points that failed validation or storage.
    """

    collector: str
    stored: int
    failed: int


class BaseCollector(ABC):
    """Template that all data collectors must follow.

    ABC stands for "Abstract Base Class" — it means you can't use
    BaseCollector directly. You have to create a specific collector
    (like PriceCollector) that fills in the abstract methods.

    Think of it like a form with blanks:
    - BaseCollector defines the form (what fields exist)
    - PriceCollector fills in the blanks (how to actually get prices)
    """

    # Each collector has a name for logging purposes.
    # Subclasses set this to something like "prices" or "economic".
    collector_name: str = "base"

    @abstractmethod
    def collect_latest(self) -> dict[str, Any]:
        """Pull the most recent data and store it.

        Every collector MUST implement this method. It should:
        1. Fetch the latest data from the source
        2. Validate through Pydantic
        3. Store valid data in the database
        4. Return a summary dict

        Returns:
            A dict with at least these keys (see CollectorResult):
            - "stored": number of data points successfully saved
            - "failed": number of data points that failed validation or storage
            - "collector": name of this collector
            Plus collector-specific extras (details, skipped_null, etc.)
        """

    @abstractmethod
    def collect_history(self, **kwargs: Any) -> dict[str, Any]:
        """Pull historical data and store it (for backfilling).

        Every collector MUST implement this method. Used when you first
        set up the system and need to load past data, or when you need
        to fill gaps.

        Args:
            **kwargs: Collector-specific options like period="1y" for
                      prices or years=1 for economic data.

        Returns:
            Same format as collect_latest().
        """

    def _build_summary(
        self,
        *,
        stored: int,
        failed: int,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build a standard summary dict that all collectors return.

        This ensures every collector's result looks the same, so the
        daily pipeline doesn't have to handle different formats.

        Args:
            stored: How many data points were saved successfully.
            failed: How many data points failed.
            **extra: Any collector-specific info to include.

        Returns:
            A dict with stored, failed, collector name, and any extras.
        """
        summary = {
            "collector": self.collector_name,
            "stored": stored,
            "failed": failed,
            **extra,
        }

        # Log a quick summary line
        if failed > 0:
            logger.warning(
                "%s collector finished: %d stored, %d failed",
                self.collector_name,
                stored,
                failed,
            )
        else:
            logger.info(
                "%s collector finished: %d stored, %d failed",
                self.collector_name,
                stored,
                failed,
            )

        return summary
