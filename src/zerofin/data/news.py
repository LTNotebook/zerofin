"""
News/RSS data collector — pulls articles from financial RSS feeds.

This module fetches news articles from ~30 must-have RSS feeds covering
financial news, economic data releases, sector-specific sources, central
banks, and international markets. Articles are stored as Article nodes
in Neo4j for later AI analysis and entity linking.

How it works:
1. Iterate through a curated list of RSS feed URLs
2. Fetch each feed's XML using httpx
3. Parse the XML (handling both RSS 2.0 and Atom formats)
4. Deduplicate against articles already in Neo4j (by URL)
5. Store new articles as Article nodes via GraphStorage.run_query()
6. Return a summary of what was collected

Why Article nodes bypass create_entity():
Article is not one of the 12 FinDKG entity types — it's a content node
that will later be connected TO entities via MENTIONED_IN edges. We use
run_query() with a direct MERGE to create Article nodes without needing
to modify ENTITY_LABELS.
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import pendulum

from zerofin.data.collector import BaseCollector
from zerofin.storage.graph import GraphStorage

# Set up logger — messages show up as "zerofin.data.news"
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
# How long to wait for a feed response before giving up (seconds).
# Some feeds are slow — 15 seconds is generous but avoids hanging forever.
FEED_TIMEOUT_SECONDS = 15

# User-agent string so feed servers know we're a bot, not a browser.
# Some feeds block requests without a user-agent header.
USER_AGENT = "Zerofin/1.0 (Financial Research Bot; +https://github.com/zerofin)"

# ── RSS Feed Registry ─────────────────────────────────────────────────
# Every feed we pull from. Each entry has:
#   name:         Human-readable name for logging
#   url:          The RSS/Atom feed URL
#   category:     What kind of news this covers (used for filtering later)
#   priority:     "must_have" or "nice_to_have"
#   content_type: How much content the feed provides:
#                 "full_text" = complete article body
#                 "summary"   = paragraph-length summary
#                 "headline_only" = just the title, no body text
#
# Only MUST-HAVE feeds are included here. Nice-to-haves can be added
# later without changing any code — just append to this list.

RSS_FEEDS: list[dict[str, str]] = [
    # ── Major Financial News ──────────────────────────────────────────
    {
        "name": "CNBC Top News",
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "category": "general",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "CNBC World News",
        "url": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "category": "international",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "CNBC Finance",
        "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "category": "general",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "CNBC Economy",
        "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "category": "economy",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "CNBC Earnings",
        "url": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "category": "earnings",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "AP News Business",
        "url": "https://apnews.com/business.rss",
        "category": "general",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "MarketWatch Top Stories",
        "url": "https://www.marketwatch.com/rss/topstories",
        "category": "general",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "MarketWatch Market Pulse",
        "url": "https://www.marketwatch.com/rss/marketpulse",
        "category": "general",
        "priority": "must_have",
        "content_type": "headline_only",
    },
    {
        "name": "MarketWatch Breaking News",
        "url": "https://www.marketwatch.com/rss/breakingnews",
        "category": "general",
        "priority": "must_have",
        "content_type": "headline_only",
    },
    {
        "name": "Yahoo Finance Headlines",
        "url": "https://finance.yahoo.com/news/rssindex",
        "category": "general",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "Yahoo Finance Tickers",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,AAPL&region=US&lang=en-US",
        "category": "general",
        "priority": "must_have",
        "content_type": "headline_only",
    },
    {
        "name": "Seeking Alpha Latest Articles",
        "url": "https://seekingalpha.com/feed.xml",
        "category": "analysis",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "Seeking Alpha Market News",
        "url": "https://seekingalpha.com/market_currents.xml",
        "category": "general",
        "priority": "must_have",
        "content_type": "headline_only",
    },
    {
        "name": "Seeking Alpha Wall St. Breakfast",
        "url": "https://seekingalpha.com/tag/wall-st-breakfast.xml",
        "category": "analysis",
        "priority": "must_have",
        "content_type": "summary",
    },
    # ── Economic Data & Government Sources ────────────────────────────
    {
        "name": "Federal Reserve All Press Releases",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Federal Reserve Monetary Policy",
        "url": "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Federal Reserve Speeches",
        "url": "https://www.federalreserve.gov/feeds/speeches.xml",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Fed Chair Powell Speeches",
        "url": "https://www.federalreserve.gov/feeds/s_t_powell.xml",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Fed H.15 Interest Rates",
        "url": "https://www.federalreserve.gov/feeds/h15.xml",
        "category": "economy",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "US Treasury Auction Announcements",
        "url": "https://www.treasurydirect.gov/TA_WS/securities/announced/rss",
        "category": "government",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "US Treasury Auction Results",
        "url": "https://www.treasurydirect.gov/TA_WS/securities/auctioned/rss",
        "category": "government",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "BLS All Updates",
        "url": "https://www.bls.gov/feed/bls_latest.rss",
        "category": "economy",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "BEA All Releases",
        "url": "https://apps.bea.gov/rss/rss.xml",
        "category": "economy",
        "priority": "must_have",
        "content_type": "summary",
    },
    # ── Sector-Specific ───────────────────────────────────────────────
    {
        "name": "EIA Today in Energy",
        "url": "https://www.eia.gov/rss/todayinenergy.xml",
        "category": "sector_energy",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "EIA Press Releases",
        "url": "https://www.eia.gov/rss/press_rss.xml",
        "category": "sector_energy",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "category": "sector_tech",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "OilPrice.com",
        "url": "https://oilprice.com/rss/main",
        "category": "sector_energy",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Defense News",
        "url": "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
        "category": "sector_defense",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "Fierce Pharma",
        "url": "https://www.fiercepharma.com/rss/xml",
        "category": "sector_healthcare",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "HousingWire",
        "url": "https://www.housingwire.com/feed/",
        "category": "sector_realestate",
        "priority": "must_have",
        "content_type": "full_text",
    },
    # ── International ─────────────────────────────────────────────────
    {
        "name": "BBC Business",
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "category": "international",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "SCMP Business",
        "url": "https://www.scmp.com/rss/92/feed",
        "category": "international",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "SCMP China Economy",
        "url": "https://www.scmp.com/rss/318421/feed",
        "category": "international",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "Nikkei Asia",
        "url": "https://asia.nikkei.com/rss/feed/nar",
        "category": "international",
        "priority": "must_have",
        "content_type": "headline_only",
    },
    {
        "name": "Al Jazeera Economy",
        "url": "https://www.aljazeera.com/xml/rss/economy.xml",
        "category": "international",
        "priority": "must_have",
        "content_type": "summary",
    },
    # ── Crypto ────────────────────────────────────────────────────────
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "category": "crypto",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "CoinTelegraph",
        "url": "https://cointelegraph.com/rss",
        "category": "crypto",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "The Block",
        "url": "https://www.theblock.co/rss.xml",
        "category": "crypto",
        "priority": "must_have",
        "content_type": "summary",
    },
    # ── Central Banks (non-Fed) ───────────────────────────────────────
    {
        "name": "ECB Press Releases",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "ECB Statistical Releases",
        "url": "https://www.ecb.europa.eu/rss/statpress.html",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "summary",
    },
    {
        "name": "Bank of Japan",
        "url": "https://www.boj.or.jp/en/rss/whatsnew.xml",
        "category": "central_bank",
        "priority": "must_have",
        "content_type": "summary",
    },
    # ── Government / Regulatory ───────────────────────────────────────
    {
        "name": "GovInfo Federal Register",
        "url": "https://www.govinfo.gov/rss/fr.xml",
        "category": "government",
        "priority": "must_have",
        "content_type": "summary",
    },
    # ── Earnings & Wire Services ──────────────────────────────────────
    {
        "name": "PR Newswire All Releases",
        "url": "https://www.prnewswire.com/rss/news-releases-list.rss",
        "category": "earnings",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Business Wire All News",
        "url": "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFtRXA==",
        "category": "earnings",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "GlobeNewsWire All Releases",
        "url": "https://www.globenewswire.com/RssFeed/feedTitle/GlobeNewswire",
        "category": "earnings",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "GlobeNewsWire Earnings",
        "url": (
            "https://www.globenewswire.com/RssFeed/subjectcode/"
            "13-Earnings Releases And Operating Results/"
            "feedTitle/GlobeNewswire - Earnings Releases And Operating Results"
        ),
        "category": "earnings",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "GlobeNewsWire M&A",
        "url": (
            "https://www.globenewswire.com/RssFeed/subjectcode/"
            "27-Mergers And Acquisitions/"
            "feedTitle/GlobeNewswire - Mergers And Acquisitions"
        ),
        "category": "earnings",
        "priority": "must_have",
        "content_type": "full_text",
    },
    # ── Alternative / Analytical ──────────────────────────────────────
    {
        "name": "Calculated Risk",
        "url": "https://feeds.feedburner.com/CalculatedRisk",
        "category": "analysis",
        "priority": "must_have",
        "content_type": "full_text",
    },
    {
        "name": "Marginal Revolution",
        "url": "https://marginalrevolution.com/feed",
        "category": "analysis",
        "priority": "must_have",
        "content_type": "full_text",
    },
]

# ── Atom namespace ────────────────────────────────────────────────────
# Atom feeds use XML namespaces. We need to tell ElementTree about them
# so it can find <entry>, <title>, etc. inside Atom documents.
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ATOM_NS_MAP = {"atom": ATOM_NAMESPACE}


# ── Helper functions ──────────────────────────────────────────────────


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from a string, leaving just the plain text.

    RSS descriptions often contain HTML markup like <p>, <a>, <b>, etc.
    We strip it all out because we want clean text for AI analysis later.
    Also unescapes HTML entities like &amp; -> & and handles CDATA.
    """
    if not text:
        return ""
    # Remove CDATA wrappers if present: <![CDATA[...]]>
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
    # Strip all HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)
    # Collapse multiple whitespace/newlines into single spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_published_date(date_string: str | None) -> str:
    """Parse a date string from an RSS feed into ISO 8601 format.

    RSS dates are notoriously inconsistent across feeds. Some use RFC 822
    (like "Mon, 23 Mar 2026 14:30:00 GMT"), some use ISO 8601, some use
    random custom formats. pendulum.parse with strict=False handles most
    of these variations.

    Returns the ISO 8601 string, or the current time if parsing fails
    (better to have an approximate time than no time at all).
    """
    if not date_string or not date_string.strip():
        return pendulum.now("UTC").to_iso8601_string()

    try:
        parsed = pendulum.parse(date_string.strip(), strict=False)
        return parsed.to_iso8601_string()
    except Exception:
        # Some dates are truly unparseable. Log it and use "now" as fallback.
        logger.debug("Could not parse date '%s', using current time", date_string)
        return pendulum.now("UTC").to_iso8601_string()


def _parse_rss_items(root: ET.Element) -> list[dict[str, str | None]]:
    """Extract articles from an RSS 2.0 feed.

    RSS 2.0 structure:
        <rss>
            <channel>
                <item>
                    <title>Headline here</title>
                    <link>https://example.com/article</link>
                    <description>Summary text...</description>
                    <pubDate>Mon, 23 Mar 2026 14:30:00 GMT</pubDate>
                </item>
                ...
            </channel>
        </rss>
    """
    items: list[dict[str, str | None]] = []

    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        description_el = item.find("description")
        pub_date_el = item.find("pubDate")

        # Skip items without a link — we use the URL as the unique ID,
        # so an article without a link is useless to us.
        link = link_el.text if link_el is not None else None
        if not link:
            continue

        items.append(
            {
                "title": _strip_html_tags(title_el.text or "") if title_el is not None else None,
                "link": link.strip(),
                "summary": (
                    _strip_html_tags(description_el.text or "")
                    if description_el is not None
                    else None
                ),
                "published": pub_date_el.text if pub_date_el is not None else None,
            }
        )

    return items


def _parse_atom_entries(root: ET.Element) -> list[dict[str, str | None]]:
    """Extract articles from an Atom feed.

    Atom structure (note the namespace):
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Headline here</title>
                <link href="https://example.com/article"/>
                <summary>Summary text...</summary>
                <published>2026-03-23T14:30:00Z</published>
            </entry>
            ...
        </feed>

    Key differences from RSS 2.0:
    - Uses <entry> instead of <item>
    - <link> has the URL in an href attribute, not as text content
    - Uses <summary> or <content> instead of <description>
    - Uses <published> or <updated> instead of <pubDate>
    """
    items: list[dict[str, str | None]] = []

    for entry in root.iter(f"{{{ATOM_NAMESPACE}}}entry"):
        title_el = entry.find(f"{{{ATOM_NAMESPACE}}}title")
        summary_el = entry.find(f"{{{ATOM_NAMESPACE}}}summary")
        published_el = entry.find(f"{{{ATOM_NAMESPACE}}}published")
        updated_el = entry.find(f"{{{ATOM_NAMESPACE}}}updated")

        # Atom stores the URL in <link href="..."> not as text content.
        # Some feeds have multiple <link> elements (e.g., alternate, self).
        # We want the one with rel="alternate" or the first one.
        link = None
        for link_el in entry.iter(f"{{{ATOM_NAMESPACE}}}link"):
            rel = link_el.get("rel", "alternate")
            if rel == "alternate":
                link = link_el.get("href")
                break
        # Fallback: just grab the first link's href if no alternate found
        if not link:
            first_link = entry.find(f"{{{ATOM_NAMESPACE}}}link")
            if first_link is not None:
                link = first_link.get("href")

        if not link:
            continue

        # Fall back to <content> if <summary> is missing
        if summary_el is None:
            summary_el = entry.find(f"{{{ATOM_NAMESPACE}}}content")

        # Use <published> if available, otherwise fall back to <updated>
        date_text = None
        if published_el is not None:
            date_text = published_el.text
        elif updated_el is not None:
            date_text = updated_el.text

        items.append(
            {
                "title": _strip_html_tags(title_el.text or "") if title_el is not None else None,
                "link": link.strip(),
                "summary": (
                    _strip_html_tags(summary_el.text or "") if summary_el is not None else None
                ),
                "published": date_text,
            }
        )

    return items


def _detect_and_parse_feed(xml_text: str) -> list[dict[str, str | None]]:
    """Detect whether the XML is RSS 2.0 or Atom, then parse accordingly.

    We check the root element's tag to figure out the format:
    - <rss> or contains <channel> -> RSS 2.0
    - <feed> with Atom namespace  -> Atom
    """
    root = ET.fromstring(xml_text)

    # Check if it's Atom (root tag contains the Atom namespace)
    if ATOM_NAMESPACE in root.tag:
        return _parse_atom_entries(root)

    # Otherwise assume RSS 2.0 (root is <rss> with <channel><item> inside)
    return _parse_rss_items(root)


# ── The Collector ─────────────────────────────────────────────────────


class NewsCollector(BaseCollector):
    """Collects news articles from RSS feeds and stores them in Neo4j.

    Each article becomes an Article node in the knowledge graph. Later,
    the AI pipeline will:
    1. Read article text
    2. Extract mentioned entities (companies, indicators, etc.)
    3. Create MENTIONED_IN edges linking entities to articles
    4. Generate embeddings for vector similarity search

    Usage:
        with GraphStorage() as graph:
            collector = NewsCollector(graph=graph)
            result = collector.collect_latest()
            print(result["stored"])  # number of new articles saved
    """

    collector_name: str = "news"

    def __init__(
        self,
        *,
        graph: GraphStorage,
        feeds: list[dict[str, str]] | None = None,
    ) -> None:
        """Set up the news collector.

        Args:
            graph: An already-connected GraphStorage instance.
            feeds: Optional list of feed configs. Defaults to RSS_FEEDS
                   (the ~30 must-have feeds defined at module level).
        """
        self._graph = graph
        self._feeds = feeds if feeds is not None else RSS_FEEDS

    def collect_latest(self) -> dict[str, Any]:
        """Fetch all RSS feeds and store new articles in Neo4j.

        Iterates through every feed, parses the articles, deduplicates
        against what's already stored, and batch-inserts new ones.

        If a single feed fails (network error, bad XML, etc.), we log
        the error and move on to the next feed. One broken feed should
        never block the rest.

        Returns:
            Summary dict with stored/failed counts and per-feed breakdown.
        """
        total_stored = 0
        total_failed = 0
        total_skipped_duplicate = 0
        feed_results: list[dict[str, Any]] = []

        for feed_config in self._feeds:
            feed_name = feed_config["name"]
            feed_url = feed_config["url"]

            try:
                result = self._process_single_feed(feed_config)
                total_stored += result["stored"]
                total_failed += result["failed"]
                total_skipped_duplicate += result["duplicates"]
                feed_results.append(result)
            except Exception:
                # Catch-all for anything unexpected. We already handle
                # network/parse errors in _process_single_feed, but this
                # catches truly unexpected issues like bugs in our code.
                logger.exception("Unexpected error processing feed '%s' (%s)", feed_name, feed_url)
                total_failed += 1
                feed_results.append(
                    {
                        "feed": feed_name,
                        "stored": 0,
                        "failed": 1,
                        "duplicates": 0,
                        "error": "unexpected_error",
                    }
                )

        return self._build_summary(
            stored=total_stored,
            failed=total_failed,
            duplicates=total_skipped_duplicate,
            feeds_processed=len(feed_results),
            feed_results=feed_results,
        )

    def collect_history(self, **kwargs: Any) -> dict[str, Any]:
        """Pull historical articles — for RSS this is the same as collect_latest.

        RSS feeds only serve recent articles (usually the last 10-50).
        There's no way to request articles from a specific date range.
        So historical collection is identical to latest collection.
        """
        logger.info("RSS feeds only provide recent articles — running collect_latest()")
        return self.collect_latest()

    # ── Private helpers ───────────────────────────────────────────────

    def _process_single_feed(self, feed_config: dict[str, str]) -> dict[str, Any]:
        """Fetch, parse, dedup, and store articles from one RSS feed.

        Args:
            feed_config: One entry from RSS_FEEDS with name, url, category, etc.

        Returns:
            Dict with stored/failed/duplicates counts for this feed.
        """
        feed_name = feed_config["name"]
        feed_url = feed_config["url"]
        category = feed_config["category"]
        content_type = feed_config["content_type"]

        logger.info("Fetching feed: %s", feed_name)

        # ── Step 1: Fetch the feed XML ────────────────────────────────
        xml_text = self._fetch_feed(feed_url, feed_name)
        if xml_text is None:
            # _fetch_feed already logged the error
            return {
                "feed": feed_name,
                "stored": 0,
                "failed": 1,
                "duplicates": 0,
                "error": "fetch_failed",
            }

        # ── Step 2: Parse the XML into article dicts ──────────────────
        try:
            raw_articles = _detect_and_parse_feed(xml_text)
        except ET.ParseError:
            logger.error("Failed to parse XML from feed '%s' (%s)", feed_name, feed_url)
            return {
                "feed": feed_name,
                "stored": 0,
                "failed": 1,
                "duplicates": 0,
                "error": "parse_failed",
            }

        if not raw_articles:
            logger.info("No articles found in feed '%s'", feed_name)
            return {"feed": feed_name, "stored": 0, "failed": 0, "duplicates": 0}

        # ── Step 3: Deduplicate against Neo4j ─────────────────────────
        # Extract all URLs from this batch, then check which ones
        # already exist in the graph. This is one query instead of N.
        article_urls = [a["link"] for a in raw_articles if a.get("link")]
        existing_urls = self._get_existing_article_urls(article_urls)

        # ── Step 4: Build Article node properties for new articles ────
        collected_at = pendulum.now("UTC").to_iso8601_string()
        new_articles: list[dict[str, Any]] = []

        for article in raw_articles:
            url = article.get("link")
            if not url or url in existing_urls:
                continue

            new_articles.append(
                {
                    "id": url,
                    "title": article.get("title") or "Untitled",
                    "url": url,
                    "summary": article.get("summary") or "",
                    "source": feed_name,
                    "source_category": category,
                    "published_date": _parse_published_date(article.get("published")),
                    "collected_at": collected_at,
                    "content_type": content_type,
                    "status": "raw",
                }
            )

        duplicates = len(raw_articles) - len(new_articles)

        if not new_articles:
            logger.info("All %d articles from '%s' already exist", len(raw_articles), feed_name)
            return {"feed": feed_name, "stored": 0, "failed": 0, "duplicates": duplicates}

        # ── Step 5: Store new articles in Neo4j ───────────────────────
        stored = self._store_articles_batch(new_articles)

        logger.info(
            "Feed '%s': %d new, %d duplicates skipped",
            feed_name,
            stored,
            duplicates,
        )

        return {"feed": feed_name, "stored": stored, "failed": 0, "duplicates": duplicates}

    def _fetch_feed(self, url: str, feed_name: str) -> str | None:
        """Fetch RSS feed XML from a URL using httpx.

        Retries once on failure (as per CLAUDE.md error handling rules).

        Args:
            url: The RSS feed URL.
            feed_name: Human-readable name for logging.

        Returns:
            The response text (XML), or None if the fetch failed.
        """
        headers = {"User-Agent": USER_AGENT}
        max_attempts = 2  # Try once, retry once on failure

        for attempt in range(1, max_attempts + 1):
            try:
                response = httpx.get(
                    url,
                    headers=headers,
                    timeout=FEED_TIMEOUT_SECONDS,
                    follow_redirects=True,
                )
                response.raise_for_status()
                return response.text
            except httpx.TimeoutException:
                logger.warning(
                    "Timeout fetching '%s' (attempt %d/%d): %s",
                    feed_name,
                    attempt,
                    max_attempts,
                    url,
                )
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %d from '%s' (attempt %d/%d): %s",
                    exc.response.status_code,
                    feed_name,
                    attempt,
                    max_attempts,
                    url,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "Request error for '%s' (attempt %d/%d): %s — %s",
                    feed_name,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    url,
                )

        # Both attempts failed
        logger.error(
            "Failed to fetch feed '%s' after %d attempts: %s",
            feed_name,
            max_attempts,
            url,
        )
        return None

    def _get_existing_article_urls(self, urls: list[str]) -> set[str]:
        """Check which article URLs already exist in Neo4j.

        Does a single Cypher query with UNWIND to check all URLs at once,
        instead of making N separate queries. Returns the set of URLs
        that are already stored so we can skip them.

        Args:
            urls: List of article URLs to check.

        Returns:
            Set of URLs that already have Article nodes in Neo4j.
        """
        if not urls:
            return set()

        # UNWIND the URL list and match against existing Article nodes.
        # Only returns URLs that have a match — missing ones are excluded.
        query = "UNWIND $urls AS url MATCH (a:Article {id: url}) RETURN a.id AS existing_url"

        try:
            records = self._graph.run_query(query, parameters={"urls": urls})
            return {record["existing_url"] for record in records}
        except Exception:
            # If the dedup check fails, log it and return empty set.
            # This means we might create some duplicates, but that's
            # better than skipping all articles or crashing.
            logger.exception("Failed to check for existing articles in Neo4j")
            return set()

    def _store_articles_batch(self, articles: list[dict[str, Any]]) -> int:
        """Store a batch of articles as Article nodes in Neo4j.

        Uses UNWIND + MERGE for efficient batch insertion. MERGE on the
        article URL ensures idempotency — running this twice with the
        same data won't create duplicates.

        Args:
            articles: List of article property dicts (each has id, title,
                      url, summary, source, etc.).

        Returns:
            Number of articles stored.
        """
        if not articles:
            return 0

        # MERGE on id (the URL) so we never create duplicate Article nodes.
        # SET n += item updates all properties if the node already exists.
        query = (
            "UNWIND $batch AS item "
            "MERGE (n:Article {id: item.id}) "
            "SET n += item "
            "RETURN count(n) AS total"
        )

        try:
            records = self._graph.run_query(query, parameters={"batch": articles})
            total = records[0]["total"] if records else 0
            logger.info("Stored %d Article nodes in Neo4j", total)
            return total
        except Exception:
            logger.exception("Failed to store articles batch in Neo4j")
            return 0


# ── Convenience function ──────────────────────────────────────────────


def collect_news(
    *,
    graph: GraphStorage | None = None,
    feeds: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Convenience function to collect news from RSS feeds.

    If no GraphStorage is provided, creates one and manages the connection
    lifecycle automatically. If you already have a graph connection open
    (e.g., in the daily pipeline), pass it in to avoid opening a second one.

    Args:
        graph: Optional pre-connected GraphStorage instance.
        feeds: Optional list of feed configs. Defaults to all must-have feeds.

    Returns:
        Summary dict from NewsCollector.collect_latest().
    """
    if graph is not None:
        # Caller provided a connection — use it directly
        collector = NewsCollector(graph=graph, feeds=feeds)
        return collector.collect_latest()

    # No connection provided — open one, collect, then close
    with GraphStorage() as managed_graph:
        collector = NewsCollector(graph=managed_graph, feeds=feeds)
        return collector.collect_latest()
