"""
FinToolsAP.WebData
===================

Plugin-based architecture for downloading WRDS data (CRSP / Compustat)
and computing stock characteristics.

Public API
----------
::

    import FinToolsAP.WebData as FTWD

    df = FTWD.getData(
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        chars=["prc", "shrout", "me", "be"],
    )

Or, for full control over the engine lifecycle::

    from FinToolsAP.WebData import WebDataEngine

    engine = WebDataEngine(username="my_wrds_user")
    df = engine.get_data(tickers=["AAPL"], chars=["me", "ret"])
    engine.close()

Extending with custom characteristics
--------------------------------------
Drop a file at ``~/.fintoolsap/custom_chars.py`` containing functions
annotated with a ``.needs`` attribute::

    def my_custom_ratio(raw_tables, freq):
        sf = raw_tables['crsp.sf']
        return sf['prc'].abs() / sf['shrout']

    my_custom_ratio.needs = {'crsp.sf': ['prc', 'shrout']}

The registry auto-discovers and registers it on import.

Architecture
------------
* :mod:`~FinToolsAP.WebData.registry` – ``Characteristic`` dataclass,
  ``CharRegistry`` singleton, plugin loader.
* :mod:`~FinToolsAP.WebData.core` – ``WebDataEngine`` orchestrator.
* :mod:`~FinToolsAP.WebData.definitions` – built-in characteristic
  definitions (CRSP, Compustat, index).
"""

from __future__ import annotations

import pandas as pd

from .registry import (
    REGISTRY,
    CharRegistry,
    Characteristic,
    characteristic,
)
from .core import WebDataEngine


# ═══════════════════════════════════════════════════════════════════════════
# Module-level convenience API
# ═══════════════════════════════════════════════════════════════════════════

_default_engine: WebDataEngine | None = None


def getData(
    tickers: list[str] | None,
    start_date=None,
    end_date=None,
    chars: list[str] | None = None,
    freq: str = "M",
    username: str | None = None,
    exchcd_filter: list[int] | None = None,
    shrcd_filter: list[int] | None = None,
) -> pd.DataFrame:
    """Module-level convenience wrapper around :meth:`WebDataEngine.get_data`.

    On first call, creates a shared engine instance using *username*.
    Subsequent calls reuse the same WRDS connection.

    Parameters
    ----------
    tickers : list[str] or None
        Stock tickers.  ``None`` → full universe.
    start_date, end_date : date-like, optional
        Date range.
    chars : list[str], optional
        Characteristics to compute.
    freq : {'M', 'D'}
        Data frequency.
    username : str, optional
        WRDS username.  Required on first call; ignored afterwards.
    exchcd_filter, shrcd_filter : list[int], optional
        Exchange / share code filters.

    Returns
    -------
    pandas.DataFrame
    """
    global _default_engine
    if _default_engine is None:
        if username is None:
            raise ValueError(
                "username is required on the first call to getData(). "
                "Example: getData(tickers=[...], username='my_wrds_user')"
            )
        _default_engine = WebDataEngine(username=username)

    return _default_engine.get_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        chars=chars,
        freq=freq,
        exchcd_filter=exchcd_filter,
        shrcd_filter=shrcd_filter,
    )


def available() -> list[str]:
    """List all registered characteristic names."""
    return REGISTRY.available()


def describe(name: str | None = None) -> dict[str, str]:
    """Return descriptions for one or all characteristics."""
    return REGISTRY.describe(name)


__all__ = [
    "WebDataEngine",
    "REGISTRY",
    "CharRegistry",
    "Characteristic",
    "characteristic",
    "getData",
    "available",
    "describe",
]
