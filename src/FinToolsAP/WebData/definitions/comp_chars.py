"""
FinToolsAP.WebData.definitions.comp_chars
==========================================

Built-in characteristics derived from Compustat Fundamentals (FUNDQ)
joined to the CRSP universe via the CCM link table.

Convention
----------
* Functions receive ``(raw_tables: dict[str, DataFrame], freq: str)``.
* ``raw_tables['comp.fundq']`` contains Compustat data already merged
  onto the CRSP panel by the engine (keyed by ``permco`` + ``date``).
* ``raw_tables['crsp.sf']`` is available when a characteristic needs
  both CRSP and Compustat inputs (e.g. book-to-market).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Compustat pass-through
# ═══════════════════════════════════════════════════════════════════════════

def atq(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Total assets (quarterly, from Compustat)."""
    panel = raw_tables["__panel__"]
    return panel["atq"]

atq.needs = {"comp.fundq": ["atq"]}
atq._output_name = "atq"
atq._order = 50


def ltq(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Total liabilities (quarterly, from Compustat)."""
    panel = raw_tables["__panel__"]
    return panel["ltq"]

ltq.needs = {"comp.fundq": ["ltq"]}
ltq._output_name = "ltq"
ltq._order = 50


# ═══════════════════════════════════════════════════════════════════════════
# Book equity
# ═══════════════════════════════════════════════════════════════════════════

def be(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Book equity = SEQ + TXDITC − coalesce(PSTKRV, PSTKQ, 0).

    Standard Fama-French definition of book equity using Compustat quarterly.
    """
    panel = raw_tables["__panel__"]

    # If the primary equity field (seqq) is NaN, the firm has no Compustat
    # coverage for this period → return NaN, not zero.
    seq = panel["seqq"]
    has_data = seq.notna()

    txditc = panel["txditcq"].fillna(0)
    pstkrq = panel["pstkrq"].astype(float)
    pstkq  = panel["pstkq"].astype(float)
    pstk = np.where(pstkrq.notna(), pstkrq, np.where(pstkq.notna(), pstkq, 0.0))
    be_raw = seq.fillna(0) + txditc - pstk

    # Mask out rows where Compustat had no data at all
    return be_raw.where(has_data, other=np.nan)

be.needs = {
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
be._output_name = "be"
be._order = 55


# ═══════════════════════════════════════════════════════════════════════════
# Earnings
# ═══════════════════════════════════════════════════════════════════════════

def earn(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Quarterly income before extraordinary items (IBQ)."""
    panel = raw_tables["__panel__"]
    return panel["ibq"]

earn.needs = {"comp.fundq": ["ibq"]}
earn._output_name = "earn"
earn._order = 55


# ═══════════════════════════════════════════════════════════════════════════
# Ratio characteristics (require both CRSP + Compustat)
# ═══════════════════════════════════════════════════════════════════════════

def bm(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Book-to-market  =  BE / ME.

    Requires ``be`` and ``me`` to have been computed first (higher _order).
    Reads from the merged panel rather than raw tables.
    """
    panel = raw_tables["__panel__"]
    book = panel["be"].astype(float)
    mkt = panel["me"].astype(float)
    result = np.where(mkt != 0, book / mkt, np.nan)
    return pd.Series(result, index=panel.index)

bm.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
bm._output_name = "bm"
bm._order = 80
bm._requires = ["be", "me"]


def bps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Book equity per share  =  BE / split-adjusted shrout."""
    panel = raw_tables["__panel__"]
    book = panel["be"].astype(float)
    shares = panel["shrout"].astype(float)
    result = np.where(shares != 0, book / shares, np.nan)
    return pd.Series(result, index=panel.index)

bps.needs = {
    "crsp.sf": ["shrout", "cfacshr"],
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
bps._output_name = "bps"
bps._order = 80
bps._requires = ["be", "shrout"]


def ep(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Earnings-to-price  =  annualized earnings / ME.

    Uses a rolling 4-quarter sum of the raw ``ibq`` column (income before
    extraordinary items) and the already-computed ``me`` characteristic.
    """
    panel = raw_tables["__panel__"]
    # Annualize earnings: rolling 4-quarter sum of raw ibq
    earn_ann = panel.groupby("permco")["ibq"].transform(
        lambda x: x.rolling(window=4, min_periods=1).sum()
    ).astype(float)
    mkt = panel["me"].astype(float)
    result = np.where(mkt != 0, earn_ann / mkt, np.nan)
    return pd.Series(result, index=panel.index)

ep.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq": ["ibq"],
}
ep._output_name = "ep"
ep._order = 85
ep._requires = ["me"]


def eps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Earnings per share  =  annualized earnings / shrout.

    Uses the raw ``ibq`` column and the already-adjusted ``shrout``
    characteristic (order 10).
    """
    panel = raw_tables["__panel__"]
    earn_ann = panel.groupby("permco")["ibq"].transform(
        lambda x: x.rolling(window=4, min_periods=1).sum()
    ).astype(float)
    shares = panel["shrout"].astype(float)
    result = np.where(shares != 0, earn_ann / shares, np.nan)
    return pd.Series(result, index=panel.index)

eps.needs = {
    "crsp.sf": ["shrout", "cfacshr"],
    "comp.fundq": ["ibq"],
}
eps._output_name = "eps"
eps._order = 85
eps._requires = ["shrout"]


# ═══════════════════════════════════════════════════════════════════════════
# abr_ead  (order 10)  —  Chan, Jegadeesh & Lakonishok (1996)
# ═══════════════════════════════════════════════════════════════════════════

def abr_ead(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Cumulative abnormal return around earnings announcement dates.

    .. math::
        \texttt{abr}_t = \sum_{\tau=-2}^{1}
            (\texttt{ret\_d}_{t+\tau} - \texttt{mkt\_d}_{t+\tau}),
        \quad \text{where } t = 0 \text{ is } \texttt{rdq}_t

    This is a quarterly variable.  Values are only populated at
    quarter-end dates; all other dates are NaN.

    Reference: Chan, Jegadeesh & Lakonishok (1996).
    """
    panel = raw_tables["__panel__"]
    engine = raw_tables.get("__engine__")

    if engine is None or engine.wrds_conn is None:
        logger.warning("abr_ead: no engine — returning NaN")
        return pd.Series(np.nan, index=panel.index)

    # rdq (report date of quarterly earnings) should be on the panel
    # after the Compustat merge.  If it's not present, return NaN.
    if "rdq" not in panel.columns:
        logger.warning("abr_ead: rdq not on panel — returning NaN")
        return pd.Series(np.nan, index=panel.index)

    # Gather unique (permco, rdq) pairs with non-null rdq
    panel_tmp = panel[["permco", "date", "rdq"]].copy()
    panel_tmp["rdq"] = pd.to_datetime(panel_tmp["rdq"], errors="coerce")
    panel_tmp = panel_tmp.dropna(subset=["rdq"])
    if panel_tmp.empty:
        return pd.Series(np.nan, index=panel.index)

    rdq_pairs = (
        panel_tmp[["permco", "rdq"]]
        .drop_duplicates()
        .copy()
    )

    dates = pd.to_datetime(panel["date"])
    start = (rdq_pairs["rdq"].min() - pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    end = (rdq_pairs["rdq"].max() + pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    permcos = rdq_pairs["permco"].dropna().astype(int).unique().tolist()
    if not permcos:
        return pd.Series(np.nan, index=panel.index)

    permco_str = ", ".join(str(p) for p in permcos)
    try:
        dsf = engine.wrds_conn.raw_sql(
            f"SELECT date, permco, ret FROM crsp.dsf "
            f"WHERE date BETWEEN '{start}' AND '{end}' "
            f"AND permco IN ({permco_str})"
        )
        dsi = engine.wrds_conn.raw_sql(
            f"SELECT date, vwretd FROM crsp.dsi "
            f"WHERE date BETWEEN '{start}' AND '{end}'"
        )
    except Exception:
        logger.error("abr_ead: DSF/DSI query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    if dsf.empty:
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")
    dsi["date"] = pd.to_datetime(dsi["date"])
    dsi["vwretd"] = pd.to_numeric(dsi["vwretd"], errors="coerce")
    dsf = dsf.merge(dsi, on="date", how="left")
    dsf = dsf.sort_values(["permco", "date"])

    # Build a trading-date index per permco for window lookups
    abr_results: dict = {}
    for pc, grp in dsf.groupby("permco"):
        grp = grp.reset_index(drop=True)
        trade_dates = grp["date"].values
        ret_arr = grp["ret"].values.astype(float)
        mkt_arr = grp["vwretd"].values.astype(float)

        pc_rdqs = rdq_pairs.loc[rdq_pairs["permco"] == pc, "rdq"].values
        for rdq_val in pc_rdqs:
            rdq_dt = pd.Timestamp(rdq_val)
            # Find the index of rdq in trading dates (or nearest)
            # Convert to numpy datetime64 so types match trade_dates
            idx = np.searchsorted(trade_dates, np.datetime64(rdq_dt))
            # Window: tau = -2 to +1 relative to event date
            lo = idx - 2
            hi = idx + 2  # exclusive upper bound (indices: idx-2, idx-1, idx, idx+1)
            if lo < 0 or hi > len(ret_arr):
                continue
            abnormal = ret_arr[lo:hi] - mkt_arr[lo:hi]
            if np.any(np.isnan(abnormal)):
                continue
            abr_results[(pc, rdq_dt)] = float(np.sum(abnormal))

    # Map results to panel rows at quarter-end dates only
    result = pd.Series(np.nan, index=panel.index)
    panel_date = pd.to_datetime(panel["date"])

    if freq == "M":
        is_quarter_end = panel_date.dt.month.isin([3, 6, 9, 12])
    else:
        # For daily: identify last trading day of each quarter
        qe = panel_date + pd.offsets.QuarterEnd(0)
        # Check if next trading day would be in a new quarter
        is_quarter_end = (panel_date == qe)

    for i in panel.index:
        if not is_quarter_end.loc[i]:
            continue
        pc = panel.loc[i, "permco"]
        rdq_val = panel_tmp.loc[panel_tmp.index.intersection([i]), "rdq"]
        if rdq_val.empty:
            # Try to find the rdq for this permco nearest to this date
            pc_rdqs = rdq_pairs.loc[rdq_pairs["permco"] == pc, "rdq"]
            p_date = panel_date.loc[i]
            # Pick the rdq in the same quarter
            q_start = p_date - pd.offsets.QuarterBegin(1, startingMonth=1)
            match = pc_rdqs[
                (pc_rdqs >= p_date - pd.DateOffset(months=3))
                & (pc_rdqs <= p_date)
            ]
            if match.empty:
                continue
            rdq_dt = pd.Timestamp(match.iloc[-1])
        else:
            rdq_dt = pd.Timestamp(rdq_val.iloc[0])
        key = (pc, rdq_dt)
        if key in abr_results:
            result.loc[i] = abr_results[key]

    return result


abr_ead.needs = {
    "crsp.sf": ["ret"],
    "comp.fundq": ["rdq"],
}
abr_ead._output_name = "abr_ead"
abr_ead._order = 10
