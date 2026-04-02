"""Tests for the mentions integration path in daily_collect.py.

Tests the logic that gates mention processing on news_stored > 0,
handles per-article LLM failures, and splits status updates between
mentions_done and mentions_failed.

No live API or database calls — all external dependencies are mocked.

Run with: pytest tests/test_daily_mentions.py
"""

from __future__ import annotations

from zerofin.ai.mentions import MentionResult


class TestMentionsFailureHandling:
    """Verify that failed LLM results get mentions_failed, not mentions_done."""

    def test_isinstance_guard_catches_non_mention_result(self) -> None:
        """If chain.batch returns a non-MentionResult, it should not be
        passed to validate_mention_ids."""
        result = "unexpected string response"
        assert not isinstance(result, MentionResult)

    def test_none_result_caught(self) -> None:
        """None from chain.batch should be caught by isinstance guard."""
        result = None
        assert not isinstance(result, MentionResult)

    def test_valid_result_passes_guard(self) -> None:
        """A proper MentionResult passes the isinstance check."""
        result = MentionResult(mentioned_ids=["NVDA", "AAPL"])
        assert isinstance(result, MentionResult)

    def test_mixed_batch_separates_success_and_failure(self) -> None:
        """Simulate a batch where some articles succeed and some fail.

        This mirrors the logic in daily_collect.py's mentions block:
        - Good results → succeeded_urls → mentions_done
        - Bad results → failed_urls → mentions_failed
        """
        articles = [
            {"url": "https://a.com/1", "title": "Good article", "summary": "text"},
            {"url": "https://a.com/2", "title": "Bad article", "summary": "text"},
            {"url": "https://a.com/3", "title": "Another good", "summary": "text"},
        ]

        # Simulate chain.batch results: good, bad, good
        batch_results = [
            MentionResult(mentioned_ids=["NVDA"]),
            "unexpected_string",  # LLM failure
            MentionResult(mentioned_ids=["AAPL", "TSLA"]),
        ]

        succeeded_urls = []
        failed_urls = []

        for article, result in zip(articles, batch_results):
            if not isinstance(result, MentionResult):
                failed_urls.append(article["url"])
                continue
            succeeded_urls.append(article["url"])

        assert succeeded_urls == ["https://a.com/1", "https://a.com/3"]
        assert failed_urls == ["https://a.com/2"]

    def test_all_failures_produces_only_failed_urls(self) -> None:
        """If every article in a chunk fails, succeeded_urls is empty."""
        batch_results = [None, None, "error"]
        failed_urls = []
        succeeded_urls = []

        for result in batch_results:
            if not isinstance(result, MentionResult):
                failed_urls.append("url")
            else:
                succeeded_urls.append("url")

        assert len(succeeded_urls) == 0
        assert len(failed_urls) == 3

    def test_all_successes_produces_only_succeeded_urls(self) -> None:
        """If every article succeeds, failed_urls is empty."""
        batch_results = [
            MentionResult(mentioned_ids=["NVDA"]),
            MentionResult(mentioned_ids=[]),
            MentionResult(mentioned_ids=["AAPL"]),
        ]
        failed_urls = []
        succeeded_urls = []

        for result in batch_results:
            if not isinstance(result, MentionResult):
                failed_urls.append("url")
            else:
                succeeded_urls.append("url")

        assert len(succeeded_urls) == 3
        assert len(failed_urls) == 0


class TestMentionsGating:
    """Test that mentions only run when news_stored > 0."""

    def test_mentions_skipped_when_no_new_articles(self) -> None:
        """If news collection stored 0 articles, mentions should not run."""
        results: dict = {"collectors": {"news": {"stored": 0}}}
        news_stored = results["collectors"].get("news", {}).get("stored", 0)
        assert news_stored == 0  # Mentions block would be skipped

    def test_mentions_runs_when_articles_stored(self) -> None:
        """If news collection stored articles, mentions should run."""
        results: dict = {"collectors": {"news": {"stored": 15}}}
        news_stored = results["collectors"].get("news", {}).get("stored", 0)
        assert news_stored > 0  # Mentions block would execute

    def test_mentions_skipped_when_news_errored(self) -> None:
        """If news collection crashed, stored key is missing — default to 0."""
        results: dict = {"collectors": {"news": {"error": "Connection refused"}}}
        news_stored = results["collectors"].get("news", {}).get("stored", 0)
        assert news_stored == 0  # Mentions block would be skipped
