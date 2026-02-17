"""
FinToolsAP.WebData.definitions.momentum_chars
===============================================

Return-based momentum and seasonality characteristics.

**All characteristics in this module are monthly-frequency only.**
Requesting them at daily frequency returns ``NaN``.

Characteristics
---------------
mom1m    short-term reversal / 1-month momentum   (order  3)
mom6m    6-month momentum                         (order  4)
mom12m   12-month momentum                        (order  5)
mom36m   36-month momentum                        (order  6)
mom60m   60-month momentum                        (order  7)
seas     return seasonality (annual lag)           (order  8)
maxret   maximum daily return in the past month   (order  9)

Convention
----------
* ``ret`` is the *raw* holding-period return from CRSP.MSF — it is **not**
  run through a separate characteristic function first.
* Cumulative-return formulas skip the current month (lag starts at τ = 1).
* ``maxret`` issues its own query to CRSP.DSF via the engine reference
  stored in ``raw_tables["__engine__"]``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _monthly_only(panel: pd.DataFrame, freq: str) -> pd.Series | None:
    """Return a NaN series if *freq* is not monthly, else ``None``."""
    if freq != "M":
        return pd.Series(np.nan, index=panel.index)
    return None


def _cum_ret(panel: pd.DataFrame, window: int, skip: int = 1) -> pd.Series:
    r"""Cumulative return over *window* periods, skipping *skip* most recent.

    .. math::
        \prod_{\tau=\text{skip}}^{\text{skip}+\text{window}-1}
        (1 + r_{t-\tau}) - 1

    Implemented via log-return summation for numerical stability.
    """
    log1r = np.log1p(panel["ret"].astype(float))
    cum = (
        panel.assign(__log1r=log1r)
        .groupby("permco")["__log1r"]
        .transform(
            lambda x: x.shift(skip).rolling(window=window, min_periods=window).sum()
        )
    )
    return np.expm1(cum)


# ═══════════════════════════════════════════════════════════════════════════
# mom1m  (order 3)  —  Jegadeesh & Titman (1993)
# ═══════════════════════════════════════════════════════════════════════════

def mom1m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Short-term reversal / 1-month momentum.

    .. math:: \texttt{mom1m}_t = \texttt{ret}_{t-1}

    Reference: Jegadeesh & Titman (1993).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard
    return panel.groupby("permco")["ret"].shift(1).astype(float)

mom1m.needs = {"crsp.sf": ["ret"]}
mom1m._output_name = "mom1m"
mom1m._order = 3


# ═══════════════════════════════════════════════════════════════════════════
# mom6m  (order 4)  —  Jegadeesh & Titman (1993)
# ═══════════════════════════════════════════════════════════════════════════

def mom6m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""6-month momentum (cumulative return months t-5 … t-1).

    .. math:: \texttt{mom6m}_t
              = \prod_{\tau=1}^{5}(1 + \texttt{ret}_{t-\tau}) - 1

    Reference: Jegadeesh & Titman (1993).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard
    return _cum_ret(panel, window=5, skip=1)

mom6m.needs = {"crsp.sf": ["ret"]}
mom6m._output_name = "mom6m"
mom6m._order = 4


# ═══════════════════════════════════════════════════════════════════════════
# mom12m  (order 5)  —  Jegadeesh (1990)
# ═══════════════════════════════════════════════════════════════════════════

def mom12m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""12-month momentum (cumulative return months t-11 … t-1).

    .. math:: \texttt{mom12m}_t
              = \prod_{\tau=1}^{11}(1 + \texttt{ret}_{t-\tau}) - 1

    Reference: Jegadeesh (1990).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard
    return _cum_ret(panel, window=11, skip=1)

mom12m.needs = {"crsp.sf": ["ret"]}
mom12m._output_name = "mom12m"
mom12m._order = 5


# ═══════════════════════════════════════════════════════════════════════════
# mom36m  (order 6)  —  Jegadeesh & Titman (1993)
# ═══════════════════════════════════════════════════════════════════════════

def mom36m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""36-month momentum (cumulative return months t-35 … t-1).

    .. math:: \texttt{mom36m}_t
              = \prod_{\tau=1}^{35}(1 + \texttt{ret}_{t-\tau}) - 1

    Reference: Jegadeesh & Titman (1993).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard
    return _cum_ret(panel, window=35, skip=1)

mom36m.needs = {"crsp.sf": ["ret"]}
mom36m._output_name = "mom36m"
mom36m._order = 6


# ═══════════════════════════════════════════════════════════════════════════
# mom60m  (order 7)  —  Jegadeesh & Titman (1993)
# ═══════════════════════════════════════════════════════════════════════════

def mom60m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""60-month momentum (cumulative return months t-59 … t-1).

    .. math:: \texttt{mom60m}_t
              = \prod_{\tau=1}^{59}(1 + \texttt{ret}_{t-\tau}) - 1

    Reference: Jegadeesh & Titman (1993).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard
    return _cum_ret(panel, window=59, skip=1)

mom60m.needs = {"crsp.sf": ["ret"]}
mom60m._output_name = "mom60m"
mom60m._order = 7


# ═══════════════════════════════════════════════════════════════════════════
# seas  (order 8)  —  Heston & Sadka (2008)
# ═══════════════════════════════════════════════════════════════════════════

def seas(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Return seasonality — the return from 12 months ago.

    .. math:: \texttt{seas}_t = \texttt{ret}_{t-12}

    Reference: Heston & Sadka (2008).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard
    return panel.groupby("permco")["ret"].shift(12).astype(float)

seas.needs = {"crsp.sf": ["ret"]}
seas._output_name = "seas"
seas._order = 8


# ═══════════════════════════════════════════════════════════════════════════
# maxret  (order 9)  —  Bali, Cakici & Whitelaw (2011)
# ═══════════════════════════════════════════════════════════════════════════

def maxret(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Maximum daily return in the most recent calendar month.

    .. math:: \texttt{maxret}_t = \max_d \texttt{ret}^d_{t}

    This characteristic requires daily returns from ``CRSP.DSF``.  It
    obtains them by issuing a separate query through the engine reference
    stored in ``raw_tables["__engine__"]``.

    Reference: Bali, Cakici & Whitelaw (2011).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only(panel, freq)
    if guard is not None:
        return guard

    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        logger.warning("maxret: no engine reference — returning NaN")
        return pd.Series(np.nan, index=panel.index)

    # Determine date range and permcos from the panel
    dates = pd.to_datetime(panel["date"])
    start_date = dates.min() - pd.DateOffset(months=1)
    end_date = dates.max()
    permcos = panel["permco"].dropna().astype(int).unique().tolist()

    if not permcos:
        return pd.Series(np.nan, index=panel.index)

    # Build and execute DSF query
    permco_str = ", ".join(str(p) for p in permcos)
    sql = (
        f"SELECT date, permco, ret "
        f"FROM crsp.dsf "
        f"WHERE date BETWEEN '{start_date.strftime('%Y-%m-%d')}' "
        f"AND '{end_date.strftime('%Y-%m-%d')}' "
        f"AND permco IN ({permco_str})"
    )
    logger.debug("maxret DSF SQL: %s", sql)
    try:
        dsf = engine.wrds_conn.raw_sql(sql)
    except Exception:
        logger.error("maxret: DSF query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    if dsf.empty:
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")

    # Compute max daily return per (permco, month-end)
    dsf["month_end"] = dsf["date"] + pd.offsets.MonthEnd(0)
    max_daily = (
        dsf.groupby(["permco", "month_end"])["ret"]
        .max()
        .reset_index()
        .rename(columns={"month_end": "date", "ret": "__maxret__"})
    )

    # Merge back to the monthly panel
    panel_dates = pd.to_datetime(panel["date"])
    panel_me = panel_dates + pd.offsets.MonthEnd(0)

    merge_key = pd.DataFrame({
        "permco": panel["permco"].values,
        "date": panel_me.values,
    })
    merge_key["permco"] = merge_key["permco"].astype("Int64")
    max_daily["permco"] = max_daily["permco"].astype("Int64")
    max_daily["date"] = pd.to_datetime(max_daily["date"])

    merged = merge_key.merge(max_daily, on=["permco", "date"], how="left")
    return pd.Series(merged["__maxret__"].values, index=panel.index)

maxret.needs = {"crsp.sf": ["ret"]}
maxret._output_name = "maxret"
maxret._order = 9
