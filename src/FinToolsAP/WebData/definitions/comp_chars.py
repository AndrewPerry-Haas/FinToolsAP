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
# Helper: restrict a series to calendar quarter-end months only
# ═══════════════════════════════════════════════════════════════════════════

def _quarter_end_only(
    series: pd.Series, panel: pd.DataFrame, freq: str,
) -> pd.Series:
    """Mask *series* so that only calendar quarter-end rows are non-NaN.

    For daily frequency the function returns all-NaN (quarterly
    characteristics are undefined at daily frequency).
    For monthly frequency, rows whose month is not in {3,6,9,12} are NaN.
    """
    if freq != "M":
        return pd.Series(np.nan, index=panel.index)
    panel_date = pd.to_datetime(panel["date"])
    is_qe = panel_date.dt.month.isin([3, 6, 9, 12])
    return series.where(is_qe, other=np.nan)


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
# adp  (order 34)  —  Chan, Lakonishok & Sougiannis (2001)
# ═══════════════════════════════════════════════════════════════════════════

def adp(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Advertising expense to price.

    .. math:: \texttt{adp}_t = \frac{\texttt{xad}_t}{\texttt{me}_t}

    ``xad`` comes from Compustat Annual (forward-filled by the engine).
    ``me`` is the monthly CRSP market equity.

    Reference: Chan, Lakonishok & Sougiannis (2001).
    """
    panel = raw_tables["__panel__"]
    xad = pd.to_numeric(panel["xad"], errors="coerce")
    mkt = panel["me"].astype(float)
    result = np.where((mkt != 0) & mkt.notna() & xad.notna(),
                      xad / mkt, np.nan)
    return pd.Series(result, index=panel.index)

adp.needs = {
    "comp.funda": ["xad"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
adp._output_name = "adp"
adp._order = 34
adp._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# at_gr  (order 35)  —  Cooper, Gulen & Schill (2008)
# ═══════════════════════════════════════════════════════════════════════════

def at_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter asset growth.

    .. math:: \texttt{at\_gr}_t
              = \frac{\texttt{atq}_t - \texttt{atq}_{t-1}}{\texttt{atq}_{t-1}}

    Quarterly-only: populated only at calendar quarter-end months;
    NaN at all other months and at daily frequency.

    Reference: Cooper, Gulen & Schill (2008).
    """
    panel = raw_tables["__panel__"]
    atq_vals = pd.to_numeric(panel["atq"], errors="coerce")
    # shift(3) on monthly panel = 1 quarter back
    atq_lag = panel.groupby("permco")["atq"].shift(3).astype(float)
    growth = np.where(
        (atq_lag != 0) & atq_lag.notna() & atq_vals.notna(),
        (atq_vals - atq_lag) / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

at_gr.needs = {"comp.fundq": ["atq"]}
at_gr._output_name = "at_gr"
at_gr._order = 35


# ═══════════════════════════════════════════════════════════════════════════
# at_gr_yoy  (order 36)  —  Cooper, Gulen & Schill (2008)
# ═══════════════════════════════════════════════════════════════════════════

def at_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year asset growth.

    .. math:: \texttt{at\_gr\_yoy}_t
              = \frac{\texttt{atq}_t - \texttt{atq}_{t-4}}{\texttt{atq}_{t-4}}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Cooper, Gulen & Schill (2008).
    """
    panel = raw_tables["__panel__"]
    atq_vals = pd.to_numeric(panel["atq"], errors="coerce")
    # shift(12) on monthly panel = 4 quarters back
    atq_lag = panel.groupby("permco")["atq"].shift(12).astype(float)
    growth = np.where(
        (atq_lag != 0) & atq_lag.notna() & atq_vals.notna(),
        (atq_vals - atq_lag) / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

at_gr_yoy.needs = {"comp.fundq": ["atq"]}
at_gr_yoy._output_name = "at_gr_yoy"
at_gr_yoy._order = 36


# ═══════════════════════════════════════════════════════════════════════════
# atl  (order 37)  —  Ortiz-Molina & Phillips (2014)
# ═══════════════════════════════════════════════════════════════════════════

def atl(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Asset liquidity.

    .. math::
        \texttt{ala}_t = \texttt{cheq}_t
            + 0.75\,(\texttt{actq}_t - \texttt{cheq}_t)
            + 0.5\,(\texttt{atq}_t - \texttt{actq}_t
                    - \texttt{gdwlq}_t - \texttt{intanq}_t)

    .. math::
        \texttt{atl}_t = \frac{\texttt{ala}_t}
            {\texttt{atq}_t + \texttt{me}_t - \texttt{ceqq}_t}

    Monthly series: quarterly Compustat fields are forward-filled by the
    engine; ``me`` updates every month.

    Reference: Ortiz-Molina & Phillips (2014).
    """
    panel = raw_tables["__panel__"]
    cheq   = pd.to_numeric(panel["cheq"],   errors="coerce").fillna(0)
    actq   = pd.to_numeric(panel["actq"],   errors="coerce")
    atq_v  = pd.to_numeric(panel["atq"],    errors="coerce")
    gdwlq  = pd.to_numeric(panel["gdwlq"],  errors="coerce").fillna(0)
    intanq = pd.to_numeric(panel["intanq"], errors="coerce").fillna(0)
    ceqq   = pd.to_numeric(panel["ceqq"],   errors="coerce")
    mkt    = panel["me"].astype(float)

    ala = cheq + 0.75 * (actq - cheq) + 0.5 * (atq_v - actq - gdwlq - intanq)
    denom = atq_v + mkt - ceqq

    result = np.where(
        (denom != 0) & denom.notna() & ala.notna(),
        ala / denom,
        np.nan,
    )
    return pd.Series(result, index=panel.index)

atl.needs = {
    "comp.fundq": ["cheq", "actq", "atq", "gdwlq", "intanq", "ceqq"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
atl._output_name = "atl"
atl._order = 37
atl._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# Book equity  (order 38)
# ═══════════════════════════════════════════════════════════════════════════

def be(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Book equity = SEQ + TXDITC − coalesce(PSTKRV, PSTKQ, 0).

    Standard Fama-French definition of book equity using Compustat
    quarterly.  Requires ``seqq > 0`` and the resulting BE must be
    positive; otherwise NaN.

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Fama & French (1992).
    """
    panel = raw_tables["__panel__"]

    seq = panel["seqq"].astype(float)
    # Require seqq > 0
    has_data = seq > 0

    txditc = panel["txditcq"].fillna(0).astype(float)
    pstkrq = panel["pstkrq"].astype(float)
    pstkq  = panel["pstkq"].astype(float)
    pstk = np.where(pstkrq.notna(), pstkrq,
                    np.where(pstkq.notna(), pstkq, 0.0))
    be_raw = seq.fillna(0) + txditc - pstk

    # Mask: no Compustat data or non-positive BE
    be_out = pd.Series(
        np.where(has_data & (be_raw > 0), be_raw, np.nan),
        index=panel.index,
    )
    return _quarter_end_only(be_out, panel, freq)

be.needs = {
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
be._output_name = "be"
be._order = 38


# ═══════════════════════════════════════════════════════════════════════════
# earn  (order 45)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def earn(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Earnings (net income).

    .. math:: \texttt{earn}_t = \texttt{niq}_t

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    return pd.to_numeric(panel["niq"], errors="coerce")

earn.needs = {"comp.fundq": ["niq"]}
earn._output_name = "earn"
earn._order = 45


# ═══════════════════════════════════════════════════════════════════════════
# cf  (order 46)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def cf(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Cashflow.

    .. math::
        \texttt{cf}_t = \begin{cases}
            \texttt{ibq}_t & \text{if } \texttt{dpq}_t \text{ is missing} \\
            \texttt{ibq}_t + \texttt{dpq}_t & \text{otherwise}
        \end{cases}

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    ibq = pd.to_numeric(panel["ibq"], errors="coerce")
    dpq = pd.to_numeric(panel["dpq"], errors="coerce")
    result = np.where(dpq.notna(), ibq + dpq, ibq)
    return pd.Series(result, index=panel.index, dtype=float)

cf.needs = {"comp.fundq": ["ibq", "dpq"]}
cf._output_name = "cf"
cf._order = 46


# ═══════════════════════════════════════════════════════════════════════════
# earn_gr  (order 47)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def earn_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter earnings growth.

    .. math:: \texttt{earn\_gr}_t
              = \frac{\texttt{earn}_t - \texttt{earn}_{t-1}}{|\texttt{earn}_{t-1}|}

    Quarterly-only: populated only at calendar quarter-end months;
    NaN at all other months and at daily frequency.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    vals = panel["earn"].astype(float)
    lag = panel.groupby("permco")["earn"].shift(3).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

earn_gr.needs = {"comp.fundq": ["niq"]}
earn_gr._output_name = "earn_gr"
earn_gr._order = 47
earn_gr._requires = ["earn"]


# ═══════════════════════════════════════════════════════════════════════════
# cf_gr  (order 48)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def cf_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter cashflow growth.

    .. math:: \texttt{cf\_gr}_t
              = \frac{\texttt{cf}_t - \texttt{cf}_{t-1}}{|\texttt{cf}_{t-1}|}

    Quarterly-only: populated only at calendar quarter-end months;
    NaN at all other months and at daily frequency.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    vals = panel["cf"].astype(float)
    lag = panel.groupby("permco")["cf"].shift(3).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

cf_gr.needs = {"comp.fundq": ["ibq", "dpq"]}
cf_gr._output_name = "cf_gr"
cf_gr._order = 48
cf_gr._requires = ["cf"]


# ═══════════════════════════════════════════════════════════════════════════
# s_gr  (order 49)  —  Lakonishok, Shleifer & Vishny (1994)
# ═══════════════════════════════════════════════════════════════════════════

def s_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter sales growth.

    .. math:: \texttt{s\_gr}_t
              = \frac{\texttt{saleq}_t - \texttt{saleq}_{t-1}}{|\texttt{saleq}_{t-1}|}

    Quarterly-only: populated only at calendar quarter-end months;
    NaN at all other months and at daily frequency.

    Reference: Lakonishok, Shleifer & Vishny (1994).
    """
    panel = raw_tables["__panel__"]
    vals = pd.to_numeric(panel["saleq"], errors="coerce")
    lag = panel.groupby("permco")["saleq"].shift(3).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

s_gr.needs = {"comp.fundq": ["saleq"]}
s_gr._output_name = "s_gr"
s_gr._order = 49


# ═══════════════════════════════════════════════════════════════════════════
# earn_gr_yoy  (order 50)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def earn_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year earnings growth.

    .. math:: \texttt{earn\_gr\_yoy}_t
              = \frac{\texttt{earn}_t - \texttt{earn}_{t-4}}{|\texttt{earn}_{t-4}|}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    vals = panel["earn"].astype(float)
    lag = panel.groupby("permco")["earn"].shift(12).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

earn_gr_yoy.needs = {"comp.fundq": ["niq"]}
earn_gr_yoy._output_name = "earn_gr_yoy"
earn_gr_yoy._order = 50
earn_gr_yoy._requires = ["earn"]


# ═══════════════════════════════════════════════════════════════════════════
# cf_gr_yoy  (order 51)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def cf_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year cashflow growth.

    .. math:: \texttt{cf\_gr\_yoy}_t
              = \frac{\texttt{cf}_t - \texttt{cf}_{t-4}}{|\texttt{cf}_{t-4}|}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    vals = panel["cf"].astype(float)
    lag = panel.groupby("permco")["cf"].shift(12).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

cf_gr_yoy.needs = {"comp.fundq": ["ibq", "dpq"]}
cf_gr_yoy._output_name = "cf_gr_yoy"
cf_gr_yoy._order = 51
cf_gr_yoy._requires = ["cf"]


# ═══════════════════════════════════════════════════════════════════════════
# s_gr_yoy  (order 52)  —  Lakonishok, Shleifer & Vishny (1994)
# ═══════════════════════════════════════════════════════════════════════════

def s_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year sales growth.

    .. math:: \texttt{s\_gr\_yoy}_t
              = \frac{\texttt{saleq}_t - \texttt{saleq}_{t-4}}{|\texttt{saleq}_{t-4}|}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Lakonishok, Shleifer & Vishny (1994).
    """
    panel = raw_tables["__panel__"]
    vals = pd.to_numeric(panel["saleq"], errors="coerce")
    lag = panel.groupby("permco")["saleq"].shift(12).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

s_gr_yoy.needs = {"comp.fundq": ["saleq"]}
s_gr_yoy._output_name = "s_gr_yoy"
s_gr_yoy._order = 52


# ═══════════════════════════════════════════════════════════════════════════
# be_gr  (order 39)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def be_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter book equity growth.

    .. math:: \texttt{be\_gr}_t = \frac{\texttt{be}_t}{\texttt{be}_{t-1}}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    be_vals = panel["be"].astype(float)
    # shift(3) on monthly panel = 1 quarter back
    be_lag = panel.groupby("permco")["be"].shift(3).astype(float)
    growth = np.where(
        (be_lag > 0) & be_lag.notna() & be_vals.notna(),
        be_vals / be_lag,
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

be_gr.needs = {
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
be_gr._output_name = "be_gr"
be_gr._order = 39
be_gr._requires = ["be"]


# ═══════════════════════════════════════════════════════════════════════════
# be_gr_yoy  (order 40)  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def be_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year book equity growth.

    .. math:: \texttt{be\_gr\_yoy}_t = \frac{\texttt{be}_t}{\texttt{be}_{t-4}}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    be_vals = panel["be"].astype(float)
    # shift(12) on monthly panel = 4 quarters back
    be_lag = panel.groupby("permco")["be"].shift(12).astype(float)
    growth = np.where(
        (be_lag > 0) & be_lag.notna() & be_vals.notna(),
        be_vals / be_lag,
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

be_gr_yoy.needs = {
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
be_gr_yoy._output_name = "be_gr_yoy"
be_gr_yoy._order = 40
be_gr_yoy._requires = ["be"]


# ═══════════════════════════════════════════════════════════════════════════
# Ratio characteristics (require both CRSP + Compustat)
# ═══════════════════════════════════════════════════════════════════════════

def bm(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Book-to-market  =  BE / ME.

    .. math:: \texttt{bm}_t = \frac{\texttt{be}_t}{\texttt{me}_t}

    Monthly series: ``be`` (quarterly-only) is forward-filled within
    each firm so the numerator updates every quarter while ``me``
    updates every month.

    Reference: Fama & French (1992).
    """
    panel = raw_tables["__panel__"]
    # Forward-fill the quarterly-only BE within each firm
    book = panel.groupby("permco")["be"].ffill().astype(float)
    mkt = panel["me"].astype(float)
    result = np.where((mkt != 0) & mkt.notna() & book.notna(),
                      book / mkt, np.nan)
    return pd.Series(result, index=panel.index)

bm.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
bm._output_name = "bm"
bm._order = 41
bm._requires = ["be", "me"]


# ═══════════════════════════════════════════════════════════════════════════
# bm_ia  (order 42)  —  Asness, Porter & Stevens (2000)
# ═══════════════════════════════════════════════════════════════════════════

def bm_ia(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Industry-adjusted book-to-market (FF-49).

    .. math:: \texttt{bm\_ia}_t
              = \texttt{bm}_t
              - \overline{\texttt{bm}_t}^{\,\text{FF-49}}

    Subtracts the cross-sectional mean of ``bm`` within each date ×
    Fama-French 49-industry group.  The industry mean is computed over
    the **full CRSP/Compustat universe** so the adjustment is meaningful
    even when only a handful of tickers are requested.

    Reference: Asness, Porter & Stevens (2000).
    """
    from .industry_chars import _map_sic_to_industry  # local to avoid circular

    panel = raw_tables["__panel__"]
    engine = raw_tables.get("__engine__")

    # ── Fallback: panel-only industry mean ──────────────────────────────
    if engine is None or engine.wrds_conn is None:
        logger.warning("bm_ia: no engine — falling back to panel-only mean")
        ind_mean = panel.groupby(["date", "ind49"])["bm"].transform("mean")
        return panel["bm"] - ind_mean

    # ── Full-universe BM query ──────────────────────────────────────────
    dates = pd.to_datetime(panel["date"])
    start_str = dates.min().strftime("%Y-%m-%d")
    end_str   = dates.max().strftime("%Y-%m-%d")

    sf_table = "crsp.msf" if freq == "M" else "crsp.dsf"
    se_table = "crsp.mseall" if freq == "M" else "crsp.dseall"

    sql = (
        f"SELECT a.date, a.permco, "
        f"ABS(a.prc) / NULLIF(a.cfacpr, 0) AS adj_prc, "
        f"a.shrout * a.cfacshr AS adj_shrout, "
        f"b.hsiccd "
        f"FROM {sf_table} a "
        f"INNER JOIN {se_table} b "
        f"ON a.permco = b.permco AND a.date = b.date "
        f"WHERE a.date BETWEEN '{start_str}' AND '{end_str}' "
        f"AND b.exchcd IN (1, 2, 3) "
        f"AND b.shrcd IN (10, 11) "
        f"AND a.prc IS NOT NULL "
        f"AND a.shrout IS NOT NULL "
        f"AND a.cfacpr IS NOT NULL AND a.cfacpr <> 0"
    )
    logger.debug("bm_ia universe SQL (CRSP): %s", sql)
    try:
        univ = engine.wrds_conn.raw_sql(sql)
    except Exception:
        logger.error("bm_ia: CRSP universe query failed — falling back",
                     exc_info=True)
        ind_mean = panel.groupby(["date", "ind49"])["bm"].transform("mean")
        return panel["bm"] - ind_mean

    if univ.empty:
        return pd.Series(np.nan, index=panel.index)

    univ["date"] = pd.to_datetime(univ["date"])
    if freq == "M":
        univ["date"] = univ["date"] + pd.offsets.MonthEnd(0)
    univ["me_univ"] = univ["adj_prc"] * univ["adj_shrout"]
    univ["ind49"] = _map_sic_to_industry(univ["hsiccd"], 49)

    # Need BE from Compustat for each permco via CCM link
    be_sql = (
        f"SELECT c.gvkey, c.datadate, "
        f"c.seqq + COALESCE(c.txditcq, 0) "
        f"- COALESCE(c.pstkrq, c.pstkq, 0) AS be_comp "
        f"FROM comp.fundq c "
        f"WHERE c.datadate BETWEEN '{start_str}' AND '{end_str}' "
        f"AND c.seqq > 0"
    )
    link_sql = (
        "SELECT gvkey, lpermco AS permco, linkdt, linkenddt "
        "FROM crsp.ccmxpf_lnkhist "
        "WHERE linktype IN ('LU', 'LC') "
        "AND linkprim IN ('P', 'C')"
    )
    logger.debug("bm_ia BE SQL: %s", be_sql)
    try:
        be_df = engine.wrds_conn.raw_sql(be_sql)
        link_df = engine.wrds_conn.raw_sql(link_sql)
    except Exception:
        logger.error("bm_ia: Compustat/link query failed — falling back",
                     exc_info=True)
        ind_mean = panel.groupby(["date", "ind49"])["bm"].transform("mean")
        return panel["bm"] - ind_mean

    if be_df.empty:
        return pd.Series(np.nan, index=panel.index)

    # Merge BE onto universe via CCM link + merge_asof
    be_df["datadate"] = pd.to_datetime(be_df["datadate"])
    be_df["be_comp"] = pd.to_numeric(be_df["be_comp"], errors="coerce")
    be_df = be_df.dropna(subset=["be_comp"])
    be_df = be_df[be_df["be_comp"] > 0]

    link_df["permco"] = link_df["permco"].astype("Int64")
    be_df = be_df.merge(link_df[["gvkey", "permco"]], on="gvkey", how="inner")
    be_df = be_df.sort_values("datadate")

    univ["permco"] = univ["permco"].astype("Int64")
    univ = univ.sort_values("date")

    univ = pd.merge_asof(
        univ,
        be_df[["permco", "datadate", "be_comp"]],
        left_on="date",
        right_on="datadate",
        by="permco",
        direction="backward",
    )

    univ["bm_univ"] = np.where(
        (univ["me_univ"] != 0) & univ["me_univ"].notna()
        & univ["be_comp"].notna(),
        univ["be_comp"] / univ["me_univ"],
        np.nan,
    )

    # Industry mean BM per (date, ind49)
    ind_mean = (
        univ.dropna(subset=["bm_univ"])
        .groupby(["date", "ind49"])["bm_univ"]
        .mean()
        .reset_index()
        .rename(columns={"bm_univ": "__ind_mean_bm__"})
    )

    # Merge onto the panel
    panel_tmp = panel[["date", "ind49", "bm"]].copy()
    panel_tmp["date"] = pd.to_datetime(panel_tmp["date"])
    if freq == "M":
        panel_tmp["date"] = panel_tmp["date"] + pd.offsets.MonthEnd(0)
    merged = panel_tmp.merge(ind_mean, on=["date", "ind49"], how="left")

    return pd.Series(
        (merged["bm"].values - merged["__ind_mean_bm__"].values),
        index=panel.index,
    )

bm_ia.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "crsp.seall": ["hsiccd"],
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
bm_ia._output_name = "bm_ia"
bm_ia._order = 42
bm_ia._requires = ["bm", "ind49"]


# ═══════════════════════════════════════════════════════════════════════════
# owc  (order 43)  —  Operating Working Capital
# ═══════════════════════════════════════════════════════════════════════════

def owc(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Operating working capital.

    .. math::
        \texttt{owc}_t = \texttt{actq}_t - \texttt{lctq}_t + \texttt{npq}_t

    ``npq`` defaults to 0 when unavailable.

    Quarterly-only: populated only at calendar quarter-end months.
    """
    panel = raw_tables["__panel__"]
    actq = pd.to_numeric(panel["actq"], errors="coerce")
    lctq = pd.to_numeric(panel["lctq"], errors="coerce")
    npq  = pd.to_numeric(panel["npq"],  errors="coerce").fillna(0)
    owc_raw = actq - lctq + npq
    return _quarter_end_only(owc_raw, panel, freq)

owc.needs = {"comp.fundq": ["actq", "lctq", "npq"]}
owc._output_name = "owc"
owc._order = 43


# ═══════════════════════════════════════════════════════════════════════════
# acc  (order 44)  —  Sloan (1996)
# ═══════════════════════════════════════════════════════════════════════════

def acc(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Accruals.

    .. math::
        \texttt{acc}_t = \frac{\texttt{owc}_t - \texttt{owc}_{t-4}}
                              {\texttt{be}_{t-4}}

    Quarterly-only: populated only at calendar quarter-end months.

    Reference: Sloan (1996).
    """
    panel = raw_tables["__panel__"]
    owc_vals = panel["owc"].astype(float)
    be_vals  = panel["be"].astype(float)
    # shift(12) = 4 quarters back on monthly panel
    owc_lag = panel.groupby("permco")["owc"].shift(12).astype(float)
    be_lag  = panel.groupby("permco")["be"].shift(12).astype(float)
    result = np.where(
        (be_lag > 0) & be_lag.notna() & owc_vals.notna() & owc_lag.notna(),
        (owc_vals - owc_lag) / be_lag,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

acc.needs = {
    "comp.fundq": ["actq", "lctq", "npq", "seqq", "txditcq", "pstkrq", "pstkq"],
}
acc._output_name = "acc"
acc._order = 44
acc._requires = ["owc", "be"]


# ═══════════════════════════════════════════════════════════════════════════
# Other ratio characteristics (require both CRSP + Compustat)
# ═══════════════════════════════════════════════════════════════════════════

def bps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Book equity per share  =  BE / split-adjusted shrout.

    Monthly series: ``be`` (quarterly-only) is forward-filled within
    each firm before dividing by monthly ``shrout``.
    """
    panel = raw_tables["__panel__"]
    # Forward-fill the quarterly-only BE within each firm
    book = panel.groupby("permco")["be"].ffill().astype(float)
    shares = panel["shrout"].astype(float)
    result = np.where((shares != 0) & shares.notna() & book.notna(),
                      book / shares, np.nan)
    return pd.Series(result, index=panel.index)

bps.needs = {
    "crsp.sf": ["shrout", "cfacshr"],
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq"],
}
bps._output_name = "bps"
bps._order = 80
bps._requires = ["be", "shrout"]


def ep(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Earnings-to-price.

    .. math:: \texttt{ep}_t
              = \frac{1}{\texttt{me}_t}
                \sum_{\tau=-3}^{0} \texttt{earn}_{t+\tau}

    Trailing four-quarter sum of ``earn`` (net income) divided by
    market equity.  Monthly series: quarterly ``earn`` is forward-filled
    before summation so that ``me`` updates every month.

    Reference: Basu (1983).
    """
    panel = raw_tables["__panel__"]
    earn_filled = panel.groupby("permco")["earn"].ffill().astype(float)
    earn_4q = (
        earn_filled
        + earn_filled.groupby(panel["permco"]).shift(3)
        + earn_filled.groupby(panel["permco"]).shift(6)
        + earn_filled.groupby(panel["permco"]).shift(9)
    )
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & earn_4q.notna(),
        earn_4q / mkt, np.nan,
    )
    return pd.Series(result, index=panel.index)

ep.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq": ["niq"],
}
ep._output_name = "ep"
ep._order = 53
ep._requires = ["earn", "me"]


# ═══════════════════════════════════════════════════════════════════════════
# cfp  (order 54)  —  Lakonishok, Shleifer & Vishny (1994)
# ═══════════════════════════════════════════════════════════════════════════

def cfp(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Cashflow-to-price.

    .. math:: \texttt{cfp}_t
              = \frac{1}{\texttt{me}_t}
                \sum_{\tau=-3}^{0} \texttt{cf}_{t+\tau}

    Trailing four-quarter sum of ``cf`` divided by market equity.

    Reference: Lakonishok, Shleifer & Vishny (1994).
    """
    panel = raw_tables["__panel__"]
    cf_filled = panel.groupby("permco")["cf"].ffill().astype(float)
    cf_4q = (
        cf_filled
        + cf_filled.groupby(panel["permco"]).shift(3)
        + cf_filled.groupby(panel["permco"]).shift(6)
        + cf_filled.groupby(panel["permco"]).shift(9)
    )
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & cf_4q.notna(),
        cf_4q / mkt, np.nan,
    )
    return pd.Series(result, index=panel.index)

cfp.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq": ["ibq", "dpq"],
}
cfp._output_name = "cfp"
cfp._order = 54
cfp._requires = ["cf", "me"]


# ═══════════════════════════════════════════════════════════════════════════
# sp  (order 55)  —  Barbee, Mukherji, & Raines (1996)
# ═══════════════════════════════════════════════════════════════════════════

def sp(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Sales-to-price.

    .. math:: \texttt{sp}_t
              = \frac{1}{\texttt{me}_t}
                \sum_{\tau=-3}^{0} \texttt{saleq}_{t+\tau}

    Trailing four-quarter sum of ``saleq`` divided by market equity.

    Reference: Barbee, Mukherji, & Raines (1996).
    """
    panel = raw_tables["__panel__"]
    sale_filled = panel.groupby("permco")["saleq"].transform(
        lambda x: pd.to_numeric(x, errors="coerce")
    ).groupby(panel["permco"]).ffill().astype(float)
    sale_4q = (
        sale_filled
        + sale_filled.groupby(panel["permco"]).shift(3)
        + sale_filled.groupby(panel["permco"]).shift(6)
        + sale_filled.groupby(panel["permco"]).shift(9)
    )
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & sale_4q.notna(),
        sale_4q / mkt, np.nan,
    )
    return pd.Series(result, index=panel.index)

sp.needs = {
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq": ["saleq"],
}
sp._output_name = "sp"
sp._order = 55
sp._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# noa  (order 56)  —  Hirshleifer, Hou, Teoh & Zhang (2004)
# ═══════════════════════════════════════════════════════════════════════════

def noa(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Net operating assets.

    .. math::
        OA_t &= \texttt{atq}_t - \texttt{cheq}_t - \texttt{ivaoq}_t \\
        OL_t &= \texttt{atq}_t - \texttt{dlcq}_t - \texttt{dlttq}_t
               - \texttt{mibq}_t - \texttt{pstkq}_t - \texttt{ceqq}_t \\
        \texttt{noa}_t &= \frac{OA_t - OL_t}{\texttt{atq}_{t-4}}

    Quarterly-only.

    Reference: Hirshleifer, Hou, Teoh & Zhang (2004).
    """
    panel = raw_tables["__panel__"]
    atq_v  = pd.to_numeric(panel["atq"],   errors="coerce")
    cheq   = pd.to_numeric(panel["cheq"],  errors="coerce").fillna(0)
    ivaoq  = pd.to_numeric(panel["ivaoq"], errors="coerce").fillna(0)
    dlcq   = pd.to_numeric(panel["dlcq"],  errors="coerce").fillna(0)
    dlttq  = pd.to_numeric(panel["dlttq"], errors="coerce").fillna(0)
    mibq   = pd.to_numeric(panel["mibq"],  errors="coerce").fillna(0)
    pstkq  = pd.to_numeric(panel["pstkq"], errors="coerce").fillna(0)
    ceqq   = pd.to_numeric(panel["ceqq"],  errors="coerce").fillna(0)

    oa = atq_v - cheq - ivaoq
    ol = atq_v - dlcq - dlttq - mibq - pstkq - ceqq
    atq_lag = panel.groupby("permco")["atq"].shift(12).astype(float)

    result = np.where(
        (atq_lag != 0) & atq_lag.notna() & oa.notna() & ol.notna(),
        (oa - ol) / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

noa.needs = {"comp.fundq": ["atq", "cheq", "ivaoq", "dlcq", "dlttq",
                             "mibq", "pstkq", "ceqq"]}
noa._output_name = "noa"
noa._order = 56


# ═══════════════════════════════════════════════════════════════════════════
# atto  (order 57)  —  Soliman (2008)
# ═══════════════════════════════════════════════════════════════════════════

def atto(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Asset turnover.

    .. math:: \texttt{atto}_t
              = \frac{\texttt{saleq}_t}{\texttt{noa}_{t-1}}

    Quarterly-only.

    Reference: Soliman (2008).
    """
    panel = raw_tables["__panel__"]
    saleq = pd.to_numeric(panel["saleq"], errors="coerce")
    noa_lag = panel.groupby("permco")["noa"].shift(3).astype(float)
    result = np.where(
        (noa_lag != 0) & noa_lag.notna() & saleq.notna(),
        saleq / noa_lag,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

atto.needs = {"comp.fundq": ["saleq", "atq", "cheq", "ivaoq", "dlcq",
                              "dlttq", "mibq", "pstkq", "ceqq"]}
atto._output_name = "atto"
atto._order = 57
atto._requires = ["noa"]


# ═══════════════════════════════════════════════════════════════════════════
# cash  (order 58)  —  Lakonishok, Shleifer & Vishny (1994)
# ═══════════════════════════════════════════════════════════════════════════

def cash(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Cash-to-assets.

    .. math:: \texttt{cash}_t
              = \frac{\texttt{cheq}_t}{\texttt{atq}_t}

    Quarterly-only.

    Reference: Lakonishok, Shleifer & Vishny (1994).
    """
    panel = raw_tables["__panel__"]
    cheq  = pd.to_numeric(panel["cheq"], errors="coerce")
    atq_v = pd.to_numeric(panel["atq"],  errors="coerce")
    result = np.where(
        (atq_v != 0) & atq_v.notna() & cheq.notna(),
        cheq / atq_v,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

cash.needs = {"comp.fundq": ["cheq", "atq"]}
cash._output_name = "cash"
cash._order = 58


# ═══════════════════════════════════════════════════════════════════════════
# cashdebt  (order 59)
# ═══════════════════════════════════════════════════════════════════════════

def cashdebt(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Cash-to-debt.

    .. math:: \texttt{cashdebt}_t
              = \frac{\texttt{cash}_t}
                     {\tfrac{1}{2}(\texttt{ltq}_t + \texttt{ltq}_{t-4})}

    Quarterly-only.
    """
    panel = raw_tables["__panel__"]
    cash_v = panel["cash"].astype(float)
    ltq_v  = pd.to_numeric(panel["ltq"], errors="coerce")
    ltq_lag = panel.groupby("permco")["ltq"].shift(12).astype(float)
    denom = 0.5 * (ltq_v + ltq_lag)
    result = np.where(
        (denom != 0) & denom.notna() & cash_v.notna(),
        cash_v / denom,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

cashdebt.needs = {"comp.fundq": ["cheq", "atq", "ltq"]}
cashdebt._output_name = "cashdebt"
cashdebt._order = 59
cashdebt._requires = ["cash"]


# ═══════════════════════════════════════════════════════════════════════════
# cfdebt  (order 60)  —  Ou & Penman (1989)
# ═══════════════════════════════════════════════════════════════════════════

def cfdebt(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Cashflow-to-debt.

    .. math:: \texttt{cfdebt}_t
              = \frac{\sum_{\tau=-3}^{0} \texttt{cf}_{t+\tau}}
                     {\tfrac{1}{2}(\texttt{ltq}_t + \texttt{ltq}_{t-4})}

    Quarterly-only.

    Reference: Ou & Penman (1989).
    """
    panel = raw_tables["__panel__"]
    cf_v = panel["cf"].astype(float)
    cf_4q = (
        cf_v
        + cf_v.groupby(panel["permco"]).shift(3)
        + cf_v.groupby(panel["permco"]).shift(6)
        + cf_v.groupby(panel["permco"]).shift(9)
    )
    ltq_v   = pd.to_numeric(panel["ltq"], errors="coerce")
    ltq_lag = panel.groupby("permco")["ltq"].shift(12).astype(float)
    denom = 0.5 * (ltq_v + ltq_lag)
    result = np.where(
        (denom != 0) & denom.notna() & cf_4q.notna(),
        cf_4q / denom,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

cfdebt.needs = {"comp.fundq": ["ibq", "dpq", "ltq"]}
cfdebt._output_name = "cfdebt"
cfdebt._order = 60
cfdebt._requires = ["cf"]


# ═══════════════════════════════════════════════════════════════════════════
# pm  (order 61)  —  Soliman (2008)
# ═══════════════════════════════════════════════════════════════════════════

def pm(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Profit margin.

    .. math:: \texttt{pm}_t
              = \frac{\texttt{oiadpq}_t}{\texttt{saleq}_t}

    Quarterly-only.

    Reference: Soliman (2008).
    """
    panel = raw_tables["__panel__"]
    oiadpq = pd.to_numeric(panel["oiadpq"], errors="coerce")
    saleq  = pd.to_numeric(panel["saleq"],  errors="coerce")
    result = np.where(
        (saleq != 0) & saleq.notna() & oiadpq.notna(),
        oiadpq / saleq,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

pm.needs = {"comp.fundq": ["oiadpq", "saleq"]}
pm._output_name = "pm"
pm._order = 61


# ═══════════════════════════════════════════════════════════════════════════
# pm_ch  (order 62)  —  Soliman (2008)
# ═══════════════════════════════════════════════════════════════════════════

def pm_ch(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Change in profit margin.

    .. math:: \texttt{pm\_ch}_t = \texttt{pm}_t - \texttt{pm}_{t-1}

    Quarterly-only.

    Reference: Soliman (2008).
    """
    panel = raw_tables["__panel__"]
    pm_v   = panel["pm"].astype(float)
    pm_lag = panel.groupby("permco")["pm"].shift(3).astype(float)
    result = np.where(
        pm_v.notna() & pm_lag.notna(),
        pm_v - pm_lag,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

pm_ch.needs = {"comp.fundq": ["oiadpq", "saleq"]}
pm_ch._output_name = "pm_ch"
pm_ch._order = 62
pm_ch._requires = ["pm"]


# ═══════════════════════════════════════════════════════════════════════════
# pm_gr  (order 63)  —  Soliman (2008)
# ═══════════════════════════════════════════════════════════════════════════

def pm_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Profit margin growth.

    .. math:: \texttt{pm\_gr}_t
              = \frac{\texttt{pm}_t - \texttt{pm}_{t-1}}{|\texttt{pm}_{t-1}|}

    Quarterly-only.

    Reference: Soliman (2008).
    """
    panel = raw_tables["__panel__"]
    pm_v   = panel["pm"].astype(float)
    pm_lag = panel.groupby("permco")["pm"].shift(3).astype(float)
    result = np.where(
        (pm_lag != 0) & pm_lag.notna() & pm_v.notna(),
        (pm_v - pm_lag) / pm_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

pm_gr.needs = {"comp.fundq": ["oiadpq", "saleq"]}
pm_gr._output_name = "pm_gr"
pm_gr._order = 63
pm_gr._requires = ["pm"]


# ═══════════════════════════════════════════════════════════════════════════
# pm_gr_yoy  (order 64)  —  Soliman (2008)
# ═══════════════════════════════════════════════════════════════════════════

def pm_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year profit margin growth.

    .. math:: \texttt{pm\_gr\_yoy}_t
              = \frac{\texttt{pm}_t - \texttt{pm}_{t-4}}{|\texttt{pm}_{t-4}|}

    Quarterly-only.

    Reference: Soliman (2008).
    """
    panel = raw_tables["__panel__"]
    pm_v   = panel["pm"].astype(float)
    pm_lag = panel.groupby("permco")["pm"].shift(12).astype(float)
    result = np.where(
        (pm_lag != 0) & pm_lag.notna() & pm_v.notna(),
        (pm_v - pm_lag) / pm_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

pm_gr_yoy.needs = {"comp.fundq": ["oiadpq", "saleq"]}
pm_gr_yoy._output_name = "pm_gr_yoy"
pm_gr_yoy._order = 64
pm_gr_yoy._requires = ["pm"]


# ═══════════════════════════════════════════════════════════════════════════
# gprft  (order 65)  —  Novy-Marx (2013)
# ═══════════════════════════════════════════════════════════════════════════

def gprft(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Gross profitability.

    .. math::
        \texttt{gprft}_t = \frac{\sum_{\tau=-3}^{0}
        (\texttt{revtq}_{t+\tau} - \texttt{cogsq}_{t+\tau})}
        {\texttt{atq}_{t-4}}

    Trailing four-quarter gross profit divided by lagged total assets.
    Quarterly-only.

    Reference: Novy-Marx (2013).
    """
    panel = raw_tables["__panel__"]
    revtq = pd.to_numeric(panel["revtq"], errors="coerce")
    cogsq = pd.to_numeric(panel["cogsq"], errors="coerce")
    gp = revtq - cogsq

    gp_4q = (
        gp
        + gp.groupby(panel["permco"]).shift(3)
        + gp.groupby(panel["permco"]).shift(6)
        + gp.groupby(panel["permco"]).shift(9)
    )
    atq_lag = panel.groupby("permco")["atq"].shift(12).astype(float)

    result = np.where(
        (atq_lag != 0) & atq_lag.notna() & gp_4q.notna(),
        gp_4q / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

gprft.needs = {"comp.fundq": ["revtq", "cogsq", "atq"]}
gprft._output_name = "gprft"
gprft._order = 65


# ═══════════════════════════════════════════════════════════════════════════
# op  (order 66)  —  Fama & French (2015)
# ═══════════════════════════════════════════════════════════════════════════

def op(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Operating profitability.

    .. math::
        \texttt{op}_t = \frac{\sum_{\tau=-3}^{0}
        (\texttt{revtq}_{t+\tau} - \texttt{cogsq}_{t+\tau}
         - \texttt{xsgaq}_{t+\tau} - \texttt{xintq}_{t+\tau})}
        {\texttt{be}_{t-4}}

    Trailing four-quarter operating income divided by lagged book equity.
    Quarterly-only.

    Reference: Fama & French (2015).
    """
    panel = raw_tables["__panel__"]
    revtq = pd.to_numeric(panel["revtq"], errors="coerce")
    cogsq = pd.to_numeric(panel["cogsq"], errors="coerce")
    xsgaq = pd.to_numeric(panel["xsgaq"], errors="coerce").fillna(0)
    xintq = pd.to_numeric(panel["xintq"], errors="coerce").fillna(0)
    oi = revtq - cogsq - xsgaq - xintq

    oi_4q = (
        oi
        + oi.groupby(panel["permco"]).shift(3)
        + oi.groupby(panel["permco"]).shift(6)
        + oi.groupby(panel["permco"]).shift(9)
    )
    be_filled = panel.groupby("permco")["be"].ffill().astype(float)
    be_lag = be_filled.groupby(panel["permco"]).shift(12)

    result = np.where(
        (be_lag != 0) & be_lag.notna() & oi_4q.notna(),
        oi_4q / be_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

op.needs = {
    "comp.fundq": ["revtq", "cogsq", "xsgaq", "xintq",
                    "seqq", "txditcq", "pstkrq", "pstkq"],
}
op._output_name = "op"
op._order = 66
op._requires = ["be"]


# ═══════════════════════════════════════════════════════════════════════════
# gprft_gr  (order 67)  —  Novy-Marx (2013)
# ═══════════════════════════════════════════════════════════════════════════

def gprft_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter gross profitability growth.

    .. math::
        \texttt{gprft\_gr}_t = \frac{\texttt{gprft}_t
        - \texttt{gprft}_{t-1}}{|\texttt{gprft}_{t-1}|}

    Quarterly-only.

    Reference: Novy-Marx (2013).
    """
    panel = raw_tables["__panel__"]
    vals = panel["gprft"].astype(float)
    lag = panel.groupby("permco")["gprft"].shift(3).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

gprft_gr.needs = {"comp.fundq": ["revtq", "cogsq", "atq"]}
gprft_gr._output_name = "gprft_gr"
gprft_gr._order = 67
gprft_gr._requires = ["gprft"]


# ═══════════════════════════════════════════════════════════════════════════
# op_gr  (order 68)  —  Novy-Marx (2013)
# ═══════════════════════════════════════════════════════════════════════════

def op_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter operating profitability growth.

    .. math::
        \texttt{op\_gr}_t = \frac{\texttt{op}_t
        - \texttt{op}_{t-1}}{|\texttt{op}_{t-1}|}

    Quarterly-only.

    Reference: Novy-Marx (2013).
    """
    panel = raw_tables["__panel__"]
    vals = panel["op"].astype(float)
    lag = panel.groupby("permco")["op"].shift(3).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

op_gr.needs = {
    "comp.fundq": ["revtq", "cogsq", "xsgaq", "xintq",
                    "seqq", "txditcq", "pstkrq", "pstkq"],
}
op_gr._output_name = "op_gr"
op_gr._order = 68
op_gr._requires = ["op"]


# ═══════════════════════════════════════════════════════════════════════════
# gprft_gr_yoy  (order 67)  —  Novy-Marx (2013)
# ═══════════════════════════════════════════════════════════════════════════

def gprft_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year gross profitability growth.

    .. math::
        \texttt{gprft\_gr\_yoy}_t = \frac{\texttt{gprft}_t
        - \texttt{gprft}_{t-4}}{|\texttt{gprft}_{t-4}|}

    Quarterly-only.

    Reference: Novy-Marx (2013).
    """
    panel = raw_tables["__panel__"]
    vals = panel["gprft"].astype(float)
    lag = panel.groupby("permco")["gprft"].shift(12).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

gprft_gr_yoy.needs = {"comp.fundq": ["revtq", "cogsq", "atq"]}
gprft_gr_yoy._output_name = "gprft_gr_yoy"
gprft_gr_yoy._order = 67
gprft_gr_yoy._requires = ["gprft"]


# ═══════════════════════════════════════════════════════════════════════════
# op_gr_yoy  (order 68)  —  Novy-Marx (2013)
# ═══════════════════════════════════════════════════════════════════════════

def op_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year operating profitability growth.

    .. math::
        \texttt{op\_gr\_yoy}_t = \frac{\texttt{op}_t
        - \texttt{op}_{t-4}}{|\texttt{op}_{t-4}|}

    Quarterly-only.

    Reference: Novy-Marx (2013).
    """
    panel = raw_tables["__panel__"]
    vals = panel["op"].astype(float)
    lag = panel.groupby("permco")["op"].shift(12).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

op_gr_yoy.needs = {
    "comp.fundq": ["revtq", "cogsq", "xsgaq", "xintq",
                    "seqq", "txditcq", "pstkrq", "pstkq"],
}
op_gr_yoy._output_name = "op_gr_yoy"
op_gr_yoy._order = 68
op_gr_yoy._requires = ["op"]


# ═══════════════════════════════════════════════════════════════════════════
# pctacc  (order 67)  —  Hafzalla, Lundholm & Van Winkle (2011)
# ═══════════════════════════════════════════════════════════════════════════

def pctacc(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Percent operating accruals.

    .. math::
        \texttt{pctacc}_t = \frac{\texttt{owc}_t - \texttt{owc}_{t-4}}
        {\left|\sum_{\tau=-3}^{0} \texttt{cf}_{t+\tau}\right|}

    Quarterly-only.

    Reference: Hafzalla, Lundholm & Van Winkle (2011).
    """
    panel = raw_tables["__panel__"]
    owc_v = panel["owc"].astype(float)
    owc_lag = panel.groupby("permco")["owc"].shift(12).astype(float)
    delta_owc = owc_v - owc_lag

    cf_filled = panel.groupby("permco")["cf"].ffill().astype(float)
    cf_4q = (
        cf_filled
        + cf_filled.groupby(panel["permco"]).shift(3)
        + cf_filled.groupby(panel["permco"]).shift(6)
        + cf_filled.groupby(panel["permco"]).shift(9)
    )
    cf_4q_abs = cf_4q.abs()

    result = np.where(
        (cf_4q_abs != 0) & cf_4q_abs.notna() & delta_owc.notna(),
        delta_owc / cf_4q_abs,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

pctacc.needs = {
    "comp.fundq": ["actq", "lctq", "npq", "ibq", "dpq"],
}
pctacc._output_name = "pctacc"
pctacc._order = 67
pctacc._requires = ["owc", "cf"]


# ═══════════════════════════════════════════════════════════════════════════
# cinvest  (order 68)  —  Titman, Wei & Xie (2004)
# ═══════════════════════════════════════════════════════════════════════════

def cinvest(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Corporate investment.

    .. math::
        \texttt{cinvest}_t = \frac{\texttt{ppentq}_t
        - \texttt{ppentq}_{t-1}}{\texttt{saleq}_t}
        - \frac{1}{3}\sum_{\tau=1}^{3}
          \frac{\texttt{ppentq}_{t-\tau}
          - \texttt{ppentq}_{t-(\tau+1)}}{\texttt{saleq}_{t-\tau}}

    Uses ``\epsilon = 0.01`` when ``saleq \le 0``.
    Quarterly-only.

    Reference: Titman, Wei & Xie (2004).
    """
    panel = raw_tables["__panel__"]
    ppentq = pd.to_numeric(panel["ppentq"], errors="coerce")
    saleq = pd.to_numeric(panel["saleq"], errors="coerce")

    eps_val = 0.01

    ppent_lag1 = ppentq.groupby(panel["permco"]).shift(3)
    ppent_lag2 = ppentq.groupby(panel["permco"]).shift(6)
    ppent_lag3 = ppentq.groupby(panel["permco"]).shift(9)
    ppent_lag4 = ppentq.groupby(panel["permco"]).shift(12)

    sale_lag1 = saleq.groupby(panel["permco"]).shift(3)
    sale_lag2 = saleq.groupby(panel["permco"]).shift(6)
    sale_lag3 = saleq.groupby(panel["permco"]).shift(9)

    # Current investment ratio
    denom_curr = np.where((saleq > 0) & saleq.notna(), saleq, eps_val)
    ratio_curr = (ppentq - ppent_lag1) / denom_curr

    # Lagged investment ratios
    denom1 = np.where((sale_lag1 > 0) & sale_lag1.notna(), sale_lag1, eps_val)
    ratio1 = (ppent_lag1 - ppent_lag2) / denom1

    denom2 = np.where((sale_lag2 > 0) & sale_lag2.notna(), sale_lag2, eps_val)
    ratio2 = (ppent_lag2 - ppent_lag3) / denom2

    denom3 = np.where((sale_lag3 > 0) & sale_lag3.notna(), sale_lag3, eps_val)
    ratio3 = (ppent_lag3 - ppent_lag4) / denom3

    avg_past = (ratio1 + ratio2 + ratio3) / 3.0
    result_raw = ratio_curr - avg_past

    # Require all components to be available
    all_ok = (
        ppentq.notna() & ppent_lag1.notna() & ppent_lag2.notna()
        & ppent_lag3.notna() & ppent_lag4.notna()
    )
    result = np.where(all_ok, result_raw, np.nan)
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

cinvest.needs = {"comp.fundq": ["ppentq", "saleq"]}
cinvest._output_name = "cinvest"
cinvest._order = 68


# ═══════════════════════════════════════════════════════════════════════════
# grltnoa  (order 69)  —  Fairfield, Whisenant & Yohn (2003)
# ═══════════════════════════════════════════════════════════════════════════

def grltnoa(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Growth in long-term net operating assets.

    .. math::
        N_t &= \texttt{rectq} + \texttt{invtq} + \texttt{ppentq}
              + \texttt{acoq} + \texttt{intanq} + \texttt{aoq}
              - (\texttt{apq} + \texttt{lcoq} + \texttt{loq}) \\
        D_t &= \Delta_{t-4}(\texttt{rectq} + \texttt{invtq} + \texttt{acoq}
              + \texttt{apq} + \texttt{lcoq})
              - \sum_{\tau=-3}^{0} \texttt{dpq}_{t+\tau} \\
        \texttt{grltnoa}_t &= \frac{\Delta_{t-4} N_t - D_t}
              {\tfrac{1}{2}(\texttt{atq}_t + \texttt{atq}_{t-4})}

    Quarterly-only.

    Reference: Fairfield, Whisenant & Yohn (2003).
    """
    panel = raw_tables["__panel__"]
    rectq  = pd.to_numeric(panel["rectq"],  errors="coerce").fillna(0)
    invtq  = pd.to_numeric(panel["invtq"],  errors="coerce").fillna(0)
    ppentq = pd.to_numeric(panel["ppentq"], errors="coerce").fillna(0)
    acoq   = pd.to_numeric(panel["acoq"],   errors="coerce").fillna(0)
    intanq = pd.to_numeric(panel["intanq"], errors="coerce").fillna(0)
    aoq    = pd.to_numeric(panel["aoq"],    errors="coerce").fillna(0)
    apq    = pd.to_numeric(panel["apq"],    errors="coerce").fillna(0)
    lcoq   = pd.to_numeric(panel["lcoq"],   errors="coerce").fillna(0)
    loq    = pd.to_numeric(panel["loq"],    errors="coerce").fillna(0)
    dpq    = pd.to_numeric(panel["dpq"],    errors="coerce").fillna(0)
    atq_v  = pd.to_numeric(panel["atq"],    errors="coerce")

    # N_t = long-term net operating assets
    N = rectq + invtq + ppentq + acoq + intanq + aoq - (apq + lcoq + loq)

    # Lagged N (4 quarters = shift 12 months)
    N_lag = N.groupby(panel["permco"]).shift(12)
    delta_N = N - N_lag

    # Working-capital items for D_t
    wc = rectq + invtq + acoq + apq + lcoq
    wc_lag = wc.groupby(panel["permco"]).shift(12)
    delta_wc = wc - wc_lag

    # Trailing four-quarter depreciation sum
    dpq_4q = (
        dpq
        + dpq.groupby(panel["permco"]).shift(3)
        + dpq.groupby(panel["permco"]).shift(6)
        + dpq.groupby(panel["permco"]).shift(9)
    )

    D = delta_wc - dpq_4q

    atq_lag = atq_v.groupby(panel["permco"]).shift(12)
    avg_at = 0.5 * (atq_v + atq_lag)

    result = np.where(
        (avg_at != 0) & avg_at.notna() & delta_N.notna() & D.notna(),
        (delta_N - D) / avg_at.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

grltnoa.needs = {
    "comp.fundq": ["rectq", "invtq", "ppentq", "acoq", "intanq",
                    "aoq", "apq", "lcoq", "loq", "dpq", "atq"],
}
grltnoa._output_name = "grltnoa"
grltnoa._order = 69


# ═══════════════════════════════════════════════════════════════════════════
# ldebt_gr  (order 70)  —  Richardson et al. (2006)
# ═══════════════════════════════════════════════════════════════════════════

def ldebt_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter long-term debt growth.

    .. math::
        \texttt{ldebt\_gr}_t = \frac{\texttt{ltq}_t
        - \texttt{ltq}_{t-1}}{\texttt{ltq}_{t-1}}

    Quarterly-only.

    Reference: Richardson et al. (2006).
    """
    panel = raw_tables["__panel__"]
    ltq_v = pd.to_numeric(panel["ltq"], errors="coerce")
    ltq_lag = ltq_v.groupby(panel["permco"]).shift(3)
    result = np.where(
        (ltq_lag != 0) & ltq_lag.notna() & ltq_v.notna(),
        (ltq_v - ltq_lag) / ltq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

ldebt_gr.needs = {"comp.fundq": ["ltq"]}
ldebt_gr._output_name = "ldebt_gr"
ldebt_gr._order = 70


# ═══════════════════════════════════════════════════════════════════════════
# ldebt_gr_yoy  (order 71)  —  Richardson et al. (2006)
# ═══════════════════════════════════════════════════════════════════════════

def ldebt_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year long-term debt growth.

    .. math::
        \texttt{ldebt\_gr\_yoy}_t = \frac{\texttt{ltq}_t
        - \texttt{ltq}_{t-4}}{\texttt{ltq}_{t-4}}

    Quarterly-only.

    Reference: Richardson et al. (2006).
    """
    panel = raw_tables["__panel__"]
    ltq_v = pd.to_numeric(panel["ltq"], errors="coerce")
    ltq_lag = ltq_v.groupby(panel["permco"]).shift(12)
    result = np.where(
        (ltq_lag != 0) & ltq_lag.notna() & ltq_v.notna(),
        (ltq_v - ltq_lag) / ltq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

ldebt_gr_yoy.needs = {"comp.fundq": ["ltq"]}
ldebt_gr_yoy._output_name = "ldebt_gr_yoy"
ldebt_gr_yoy._order = 71


# ═══════════════════════════════════════════════════════════════════════════
# lev  (order 72)  —  Bhandari (1988)
# ═══════════════════════════════════════════════════════════════════════════

def lev(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Leverage  =  total liabilities / market equity.

    .. math:: \texttt{lev}_t = \frac{\texttt{ltq}_t}{\texttt{me}_t}

    Monthly series: ``ltq`` (quarterly-only) is forward-filled within
    each firm so the numerator updates every quarter while ``me``
    updates every month.

    Reference: Bhandari (1988).
    """
    panel = raw_tables["__panel__"]
    ltq_filled = panel.groupby("permco")["ltq"].ffill().astype(float)
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & ltq_filled.notna(),
        ltq_filled / mkt,
        np.nan,
    )
    return pd.Series(result, index=panel.index)

lev.needs = {
    "comp.fundq": ["ltq"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
lev._output_name = "lev"
lev._order = 72
lev._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# tx_gr  (order 73)  —  Thomas & Zhang (2011)
# ═══════════════════════════════════════════════════════════════════════════

def tx_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter tax expense growth.

    .. math::
        \texttt{tx\_gr}_t = \frac{\texttt{txtq}_t
        - \texttt{txtq}_{t-1}}{\texttt{atq}_{t-1}}

    Quarterly-only.

    Reference: Thomas & Zhang (2011).
    """
    panel = raw_tables["__panel__"]
    txtq = pd.to_numeric(panel["txtq"], errors="coerce")
    atq_v = pd.to_numeric(panel["atq"], errors="coerce")
    txtq_lag = txtq.groupby(panel["permco"]).shift(3)
    atq_lag = atq_v.groupby(panel["permco"]).shift(3)
    result = np.where(
        (atq_lag != 0) & atq_lag.notna() & txtq.notna() & txtq_lag.notna(),
        (txtq - txtq_lag) / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

tx_gr.needs = {"comp.fundq": ["txtq", "atq"]}
tx_gr._output_name = "tx_gr"
tx_gr._order = 73


# ═══════════════════════════════════════════════════════════════════════════
# tx_gr_yoy  (order 74)  —  Thomas & Zhang (2011)
# ═══════════════════════════════════════════════════════════════════════════

def tx_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year tax expense growth.

    .. math::
        \texttt{tx\_gr\_yoy}_t = \frac{\texttt{txtq}_t
        - \texttt{txtq}_{t-4}}{\texttt{atq}_{t-4}}

    Quarterly-only.

    Reference: Thomas & Zhang (2011).
    """
    panel = raw_tables["__panel__"]
    txtq = pd.to_numeric(panel["txtq"], errors="coerce")
    atq_v = pd.to_numeric(panel["atq"], errors="coerce")
    txtq_lag = txtq.groupby(panel["permco"]).shift(12)
    atq_lag = atq_v.groupby(panel["permco"]).shift(12)
    result = np.where(
        (atq_lag != 0) & atq_lag.notna() & txtq.notna() & txtq_lag.notna(),
        (txtq - txtq_lag) / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

tx_gr_yoy.needs = {"comp.fundq": ["txtq", "atq"]}
tx_gr_yoy._output_name = "tx_gr_yoy"
tx_gr_yoy._order = 74


# ═══════════════════════════════════════════════════════════════════════════
# depr  (order 75)  —  Holthausen & Larcker (1992)
# ═══════════════════════════════════════════════════════════════════════════

def depr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Depreciation to PP&E.

    .. math::
        \texttt{depr}_t = \frac{\sum_{\tau=-3}^{0}
        \texttt{dpq}_{t+\tau}}{\texttt{ppentq}_t}

    Trailing four-quarter depreciation divided by current PP&E.
    Quarterly-only.

    Reference: Holthausen & Larcker (1992).
    """
    panel = raw_tables["__panel__"]
    dpq = pd.to_numeric(panel["dpq"], errors="coerce").fillna(0)
    ppentq = pd.to_numeric(panel["ppentq"], errors="coerce")

    dpq_4q = (
        dpq
        + dpq.groupby(panel["permco"]).shift(3)
        + dpq.groupby(panel["permco"]).shift(6)
        + dpq.groupby(panel["permco"]).shift(9)
    )

    result = np.where(
        (ppentq != 0) & ppentq.notna() & dpq_4q.notna(),
        dpq_4q / ppentq.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

depr.needs = {"comp.fundq": ["dpq", "ppentq"]}
depr._output_name = "depr"
depr._order = 75


# ═══════════════════════════════════════════════════════════════════════════
# rdsale  (order 76)  —  Guo, Lev & Shi (2006)
# ═══════════════════════════════════════════════════════════════════════════

def rdsale(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""R&D-to-sales.

    .. math::
        \texttt{rdsale}_t = \frac{\sum_{\tau=-3}^{0}
        \texttt{xrdq}_{t+\tau}}{\sum_{\tau=-3}^{0}
        \texttt{saleq}_{t+\tau}}

    Trailing four-quarter R&D divided by trailing four-quarter sales.
    Quarterly-only.

    Reference: Guo, Lev & Shi (2006).
    """
    panel = raw_tables["__panel__"]
    xrdq = pd.to_numeric(panel["xrdq"], errors="coerce").fillna(0)
    saleq = pd.to_numeric(panel["saleq"], errors="coerce")

    xrd_4q = (
        xrdq
        + xrdq.groupby(panel["permco"]).shift(3)
        + xrdq.groupby(panel["permco"]).shift(6)
        + xrdq.groupby(panel["permco"]).shift(9)
    )
    sale_4q = (
        saleq
        + saleq.groupby(panel["permco"]).shift(3)
        + saleq.groupby(panel["permco"]).shift(6)
        + saleq.groupby(panel["permco"]).shift(9)
    )

    result = np.where(
        (sale_4q != 0) & sale_4q.notna() & xrd_4q.notna(),
        xrd_4q / sale_4q.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

rdsale.needs = {"comp.fundq": ["xrdq", "saleq"]}
rdsale._output_name = "rdsale"
rdsale._order = 76


# ═══════════════════════════════════════════════════════════════════════════
# rdm  (order 77)  —  Chan, Lakonishok & Sougiannis (2001)
# ═══════════════════════════════════════════════════════════════════════════

def rdm(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""R&D-to-market.

    .. math::
        \texttt{rdm}_t = \frac{1}{\texttt{me}_t}
        \sum_{\tau=-3}^{0} \texttt{xrdq}_{t+\tau}

    Trailing four-quarter R&D expense divided by market equity.
    Monthly series: quarterly ``xrdq`` is forward-filled before
    summation so that ``me`` updates every month.

    Reference: Chan, Lakonishok & Sougiannis (2001).
    """
    panel = raw_tables["__panel__"]
    xrdq = pd.to_numeric(panel["xrdq"], errors="coerce").fillna(0)
    xrd_filled = xrdq.groupby(panel["permco"]).ffill()
    xrd_4q = (
        xrd_filled
        + xrd_filled.groupby(panel["permco"]).shift(3)
        + xrd_filled.groupby(panel["permco"]).shift(6)
        + xrd_filled.groupby(panel["permco"]).shift(9)
    )
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & xrd_4q.notna(),
        xrd_4q / mkt,
        np.nan,
    )
    return pd.Series(result, index=panel.index)

rdm.needs = {
    "comp.fundq": ["xrdq"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
rdm._output_name = "rdm"
rdm._order = 77
rdm._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# rna  (order 78)  —  Soliman (2008)
# ═══════════════════════════════════════════════════════════════════════════

def rna(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Return on net operating assets.

    .. math::
        \texttt{rna}_t = \frac{\texttt{oiadpq}_t}{\texttt{noa}_{t-4}}

    Quarterly-only.

    Reference: Soliman (2008).
    """
    panel = raw_tables["__panel__"]
    oiadpq = pd.to_numeric(panel["oiadpq"], errors="coerce")
    noa_filled = panel.groupby("permco")["noa"].ffill().astype(float)
    noa_lag = noa_filled.groupby(panel["permco"]).shift(12)

    result = np.where(
        (noa_lag != 0) & noa_lag.notna() & oiadpq.notna(),
        oiadpq / noa_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

rna.needs = {
    "comp.fundq": ["oiadpq", "atq", "cheq", "ivaoq", "dlcq",
                    "dlttq", "mibq", "pstkq", "ceqq"],
}
rna._output_name = "rna"
rna._order = 78
rna._requires = ["noa"]


# ═══════════════════════════════════════════════════════════════════════════
# roa  (order 79)  —  Balakrishnan, Bartov & Faurel (2010)
# ═══════════════════════════════════════════════════════════════════════════

def roa(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Return on assets.

    .. math::
        \texttt{roa}_t = \frac{\texttt{ibq}_t}{\texttt{atq}_{t-1}}

    Quarterly-only.

    Reference: Balakrishnan, Bartov & Faurel (2010).
    """
    panel = raw_tables["__panel__"]
    ibq = pd.to_numeric(panel["ibq"], errors="coerce")
    atq_v = pd.to_numeric(panel["atq"], errors="coerce")
    atq_lag = atq_v.groupby(panel["permco"]).shift(3)
    result = np.where(
        (atq_lag != 0) & atq_lag.notna() & ibq.notna(),
        ibq / atq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

roa.needs = {"comp.fundq": ["ibq", "atq"]}
roa._output_name = "roa"
roa._order = 79


# ═══════════════════════════════════════════════════════════════════════════
# roe  (order 80)  —  Hou, Xue & Zhang (2014)
# ═══════════════════════════════════════════════════════════════════════════

def roe(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Return on equity.

    .. math::
        \texttt{roe}_t = \frac{\texttt{ibq}_t}{\texttt{ceqq}_{t-1}}

    Quarterly-only.

    Reference: Hou, Xue & Zhang (2014).
    """
    panel = raw_tables["__panel__"]
    ibq = pd.to_numeric(panel["ibq"], errors="coerce")
    ceqq = pd.to_numeric(panel["ceqq"], errors="coerce")
    ceqq_lag = ceqq.groupby(panel["permco"]).shift(3)
    result = np.where(
        (ceqq_lag != 0) & ceqq_lag.notna() & ibq.notna(),
        ibq / ceqq_lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

roe.needs = {"comp.fundq": ["ibq", "ceqq"]}
roe._output_name = "roe"
roe._order = 80


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
