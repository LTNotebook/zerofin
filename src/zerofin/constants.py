"""
Shared constants used across multiple modules.

This module prevents circular imports by providing a neutral location for
constants that both models/ and storage/ need. For example, ENTITY_LABELS
is used by both entities.py (validation) and graph.py (index setup).
"""

from __future__ import annotations

# ── Every node label the system supports ──────────────────────────────
# These map to the 12 entity types from the FinDKG ontology.
# If you add a new entity type, add it here AND re-run setup_indexes().
ENTITY_LABELS: list[str] = [
    "Asset",
    "Indicator",
    "Sector",
    "Event",
    "Company",
    "Index",
    "Commodity",
    "Currency",
    "Country",
    "CentralBank",
    "GovernmentBody",
    "Person",
]
