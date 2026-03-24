"""
Entry point for the Zerofin financial intelligence system.

This file is a placeholder for now. Eventually this will wire together
the daily pipeline, the web server, or a CLI — whichever the user invokes.
"""

from __future__ import annotations

import logging

# Set up basic logging for top-level entry point invocations.
# The worker and web app configure their own logging more precisely —
# this is just a sensible default when running main.py directly.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Placeholder entry point — will launch the pipeline or web app."""
    logger.info("Zerofin starting up")


if __name__ == "__main__":
    main()
