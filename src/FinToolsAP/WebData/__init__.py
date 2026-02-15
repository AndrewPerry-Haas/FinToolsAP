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

from typing import Union

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
    tickers: list[str] | None = None,
    permcos: list[int | str] | None = None,
    permnos: list[int | str] | None = None,
    cusips: list[str] | None = None,
    gvkeys: list[str] | None = None,
    start_date=None,
    end_date=None,
    chars: list[str] | None = None,
    freq: str = "M",
    username: str | None = None,
    exchcd_filter: list[int] | None = None,
    shrcd_filter: list[int] | None = None,
    ff_dataset: str | None = None,
) -> Union[pd.DataFrame, dict]:
    """Module-level convenience wrapper around :meth:`WebDataEngine.get_data`.

    On first call, creates a shared engine instance using *username*.
    Subsequent calls reuse the same WRDS connection.

    Parameters
    ----------
    tickers : list[str] or None
        Stock tickers (CRSP).
    permcos : list[int | str] or None
        CRSP permanent company identifiers.
    permnos : list[int | str] or None
        CRSP permanent security identifiers.
    cusips : list[str] or None
        CUSIP identifiers (matched via CRSP).
    gvkeys : list[str] or None
        Compustat Global Company Keys.
    start_date, end_date : date-like, optional
        Date range.
    chars : list[str], optional
        Characteristics to compute.  Use ``'ibes.fpi1'``,
        ``'ibes.fpi3'``, etc. for IBES forecast-period-specific
        consensus EPS estimates.
    freq : {'M', 'D'}
        Data frequency.
    username : str, optional
        WRDS username.  Required on first call; ignored afterwards.
    exchcd_filter, shrcd_filter : list[int], optional
        Exchange / share code filters.
    ff_dataset : str, optional
        Name of a Ken French data library dataset to fetch via
        ``pandas_datareader`` (e.g. ``'F-F_Research_Data_Factors'``).
        **Cannot** be used simultaneously with WRDS identifiers.
        When provided the return value is a ``dict``.

    Returns
    -------
    pandas.DataFrame or dict
        DataFrame for WRDS queries; dict for Fama-French queries.
    """
    global _default_engine
    if _default_engine is None:
        if ff_dataset is not None and username is None:
            # Fama-French only – create a minimal engine.
            # The engine constructor requires a WRDS username, but for
            # FF-only queries we bypass WRDS entirely.  We still need
            # an engine instance to call get_data(), so we use a
            # lightweight path that defers the WRDS connection.
            pass  # fall through – get_data will return before WRDS is used
        elif username is None:
            raise ValueError(
                "username is required on the first call to getData(). "
                "Example: getData(tickers=[...], username='my_wrds_user')"
            )

    # For Fama-French-only queries, bypass the engine entirely
    if ff_dataset is not None:
        try:
            import pandas_datareader.data as pdr
        except ImportError:
            raise ImportError(
                "pandas_datareader is required for Fama-French data. "
                "Install it with: pip install pandas-datareader"
            )
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else None
        return pdr.DataReader(ff_dataset, "famafrench", start=start, end=end)

    if _default_engine is None:
        _default_engine = WebDataEngine(username=username)

    return _default_engine.get_data(
        tickers=tickers,
        permcos=permcos,
        permnos=permnos,
        cusips=cusips,
        gvkeys=gvkeys,
        start_date=start_date,
        end_date=end_date,
        chars=chars,
        freq=freq,
        exchcd_filter=exchcd_filter,
        shrcd_filter=shrcd_filter,
        ff_dataset=ff_dataset,
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
