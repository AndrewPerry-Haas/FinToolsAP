"""
FinToolsAP.WebData.definitions.ibes_chars
==========================================

Built-in characteristics derived from IBES (Institutional Brokers'
Estimate System) detail history files.

The engine fetches IBES data from ``IBES.DET_EPSUS`` and cleans the
raw analyst-level records (date alignment, CUSIP trimming) in the merge
zone (``core.py`` step 4f).  Aggregation logic lives here so that each
IBES-based characteristic can choose its own method (mean, median, etc.).

For flexible forecast-period-indicator (fpi) queries, users should
request ``ibes.fpi1``, ``ibes.fpi3``, etc. in the ``chars`` list.
Those are handled dynamically in ``core.py`` via ``_make_ibes_fpi_char``
and do **not** need to be defined here.

The ``eps_est`` characteristic below uses the default fpi=1 and is
provided as a convenience for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Consensus EPS estimate (fpi = 1, backward-compatible default)
# ═══════════════════════════════════════════════════════════════════════════

def eps_est(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Consensus (mean) EPS estimate from IBES detail file (fpi = 1).

    Reads the cleaned (but unaggregated) analyst-level IBES table,
    filters to ``fpi == '1'`` (one-year forecast), computes the mean
    ``value`` per ``(date, cusip)``, and maps the result onto the CRSP
    panel via CUSIP.

    For other forecast horizons, use ``ibes.fpi3`` etc. in the chars
    list instead; those are handled dynamically by the engine.
    """
    panel = raw_tables["__panel__"]
    ibes = raw_tables.get("ibes.det_epsus")

    if ibes is None or ibes.empty or "cusip" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)

    # Filter to fpi = 1 (one-year-ahead forecast)
    ibes_fpi1 = ibes[ibes["fpi"].astype(str) == "1"]
    if ibes_fpi1.empty:
        return pd.Series(np.nan, index=panel.index)

    # Consensus: mean analyst estimate per (date, cusip)
    consensus = (
        ibes_fpi1.groupby(["date", "cusip"])[["value"]]
        .mean()
        .reset_index()
        .rename(columns={"value": "eps_est"})
    )

    # Merge onto panel
    merged = panel[["date", "cusip"]].merge(
        consensus, on=["date", "cusip"], how="left",
    )
    return merged["eps_est"]


eps_est.needs = {"ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"]}
eps_est._output_name = "eps_est"
eps_est._order = 60
