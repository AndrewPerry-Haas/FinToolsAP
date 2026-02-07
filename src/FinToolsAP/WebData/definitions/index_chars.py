"""
FinToolsAP.WebData.definitions.index_chars
===========================================

Built-in characteristics from CRSP market-level index files
(MSI / DSI).

These are not per-stock characteristics but market-wide series.
The engine merges them onto the panel by ``date``.
"""

from __future__ import annotations

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# CRSP index series
# ═══════════════════════════════════════════════════════════════════════════

def spindx(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """S&P 500 composite index level."""
    panel = raw_tables["__panel__"]
    return panel["spindx"]

spindx.needs = {"crsp.si": ["spindx"]}
spindx._output_name = "spindx"
spindx._order = 5


def sprtrn(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """S&P 500 composite total return."""
    panel = raw_tables["__panel__"]
    return panel["sprtrn"]

sprtrn.needs = {"crsp.si": ["sprtrn"]}
sprtrn._output_name = "sprtrn"
sprtrn._order = 5


def vwretd(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """CRSP value-weighted index return (with dividends)."""
    panel = raw_tables["__panel__"]
    return panel["vwretd"]

vwretd.needs = {"crsp.si": ["vwretd"]}
vwretd._output_name = "vwretd"
vwretd._order = 5


def vwretx(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """CRSP value-weighted index return (ex dividends)."""
    panel = raw_tables["__panel__"]
    return panel["vwretx"]

vwretx.needs = {"crsp.si": ["vwretx"]}
vwretx._output_name = "vwretx"
vwretx._order = 5


def ewretd(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """CRSP equal-weighted index return (with dividends)."""
    panel = raw_tables["__panel__"]
    return panel["ewretd"]

ewretd.needs = {"crsp.si": ["ewretd"]}
ewretd._output_name = "ewretd"
ewretd._order = 5


def ewretx(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """CRSP equal-weighted index return (ex dividends)."""
    panel = raw_tables["__panel__"]
    return panel["ewretx"]

ewretx.needs = {"crsp.si": ["ewretx"]}
ewretx._output_name = "ewretx"
ewretx._order = 5


def totval(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Total market value of CRSP indices."""
    panel = raw_tables["__panel__"]
    return panel["totval"]

totval.needs = {"crsp.si": ["totval"]}
totval._output_name = "totval"
totval._order = 5


def totcnt(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Number of securities in CRSP indices."""
    panel = raw_tables["__panel__"]
    return panel["totcnt"]

totcnt.needs = {"crsp.si": ["totcnt"]}
totcnt._output_name = "totcnt"
totcnt._order = 5
