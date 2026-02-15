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

import numpy as np
import pandas as pd


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
