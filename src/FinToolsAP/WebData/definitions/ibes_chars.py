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

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# IBES consensus helper
# ═══════════════════════════════════════════════════════════════════════════

def _ibes_consensus(
    raw_tables: dict[str, pd.DataFrame],
    fpi: int | str,
    col_name: str,
) -> pd.Series:
    """Compute mean consensus EPS for a given *fpi* and merge onto panel.

    Returns a Series aligned to ``panel.index`` with column name *col_name*.
    """
    panel = raw_tables["__panel__"]
    ibes = raw_tables.get("ibes.det_epsus")

    if ibes is None or ibes.empty or "cusip" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)

    ibes_sub = ibes[ibes["fpi"].astype(str) == str(fpi)]
    if ibes_sub.empty:
        return pd.Series(np.nan, index=panel.index)

    consensus = (
        ibes_sub.groupby(["date", "cusip"])[["value"]]
        .mean()
        .reset_index()
        .rename(columns={"value": col_name})
    )
    merged = panel[["date", "cusip"]].merge(
        consensus, on=["date", "cusip"], how="left",
    )
    return merged[col_name]


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
eps_est._ibes_fpi = {1}


# ═══════════════════════════════════════════════════════════════════════════
# re  (order 94)  —  Analyst Earnings Forecast Revisions  (Chan 1996)
# ═══════════════════════════════════════════════════════════════════════════

def re(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Analyst earnings forecast revisions.

    .. math::
        \texttt{re}_t = \sum_{\tau=0}^{6}
        \frac{\texttt{shrout}_{t-\tau}\,\texttt{eps1}_{t-\tau}
            - \texttt{shrout}_{t-1-\tau}\,\texttt{eps1}_{t-1-\tau}}
             {\texttt{prc}_{t-1-\tau}}

    Reference: Chan, Jegadeesh & Lakonishok (1996).
    """
    panel = raw_tables["__panel__"]
    eps1 = _ibes_consensus(raw_tables, fpi=1, col_name="__eps1__")

    shrout_vals = panel["shrout"].astype(float)
    prc_vals = panel["prc"].astype(float)

    product = shrout_vals * eps1  # shrout * eps1

    result = pd.Series(0.0, index=panel.index, dtype=float)
    for tau in range(7):
        prod_lag  = panel.assign(__p=product).groupby("permco")["__p"].shift(tau).astype(float)
        prod_lag1 = panel.assign(__p=product).groupby("permco")["__p"].shift(tau + 1).astype(float)
        prc_lag1  = panel.assign(__p=prc_vals).groupby("permco")["__p"].shift(tau + 1).astype(float)
        term = np.where(
            (prc_lag1 != 0) & prc_lag1.notna() & prod_lag.notna() & prod_lag1.notna(),
            (prod_lag - prod_lag1) / prc_lag1.abs(),
            np.nan,
        )
        result = result + pd.Series(term, index=panel.index, dtype=float)

    # If all terms were NaN, result should be NaN (not 0)
    all_nan_mask = result == 0.0
    for tau in range(7):
        prod_lag  = panel.assign(__p=product).groupby("permco")["__p"].shift(tau).astype(float)
        all_nan_mask = all_nan_mask & prod_lag.isna()
    result = result.where(~all_nan_mask, other=np.nan)

    return result

re.needs = {
    "ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
re._output_name = "re"
re._order = 94
re._requires = ["prc", "shrout"]
re._ibes_fpi = {1}


# ═══════════════════════════════════════════════════════════════════════════
# ep1  (order 95)  —  Forward Earnings-to-Price  (Lettau & Ludvigson 2018)
# ═══════════════════════════════════════════════════════════════════════════

def ep1(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Forward earnings-to-price ratio.

    .. math:: \texttt{ep1}_t = \frac{\texttt{eps1}_t}{\texttt{prc}_t}

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    eps1 = _ibes_consensus(raw_tables, fpi=1, col_name="__eps1__")
    prc_vals = panel["prc"].astype(float)

    result = np.where(
        (prc_vals != 0) & prc_vals.notna() & eps1.notna(),
        eps1 / prc_vals,
        np.nan,
    )
    return pd.Series(result, index=panel.index, dtype=float)

ep1.needs = {
    "ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"],
    "crsp.sf": ["prc", "cfacpr"],
}
ep1._output_name = "ep1"
ep1._order = 95
ep1._requires = ["prc"]
ep1._ibes_fpi = {1}


# ═══════════════════════════════════════════════════════════════════════════
# eltg  (order 96)  —  Expected LT Earnings Growth
#                       (Lettau & Ludvigson 2018)
# ═══════════════════════════════════════════════════════════════════════════

def eltg(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Expected long-term earnings growth (3-year IBES forecast).

    .. math:: \texttt{eltg}_t = \frac{\texttt{eps3}_t - \texttt{eps}_t}{\texttt{eps}_t}

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    eps3 = _ibes_consensus(raw_tables, fpi=3, col_name="__eps3__")
    eps_vals = panel["eps"].astype(float) if "eps" in panel.columns else pd.Series(np.nan, index=panel.index)

    result = np.where(
        eps3.notna() & eps_vals.notna(),
        (eps3 - eps_vals) / eps_vals.replace({0: np.nan}),
        np.nan,
    )
    return pd.Series(result, index=panel.index, dtype=float)

eltg.needs = {
    "ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"],
    "comp.fundq": ["niq"],
    "crsp.sf": ["shrout", "cfacshr"],
}
eltg._output_name = "eltg"
eltg._order = 96
eltg._requires = ["eps"]
eltg._ibes_fpi = {3}



