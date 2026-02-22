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


# ═══════════════════════════════════════════════════════════════════════════
# indcon  (order 81)  —  Hou & Robinson (2006)
# ═══════════════════════════════════════════════════════════════════════════

def indcon(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Industry concentration (Herfindahl index of sales within FF-49).

    .. math::
        \texttt{indcon}_t = \frac{1}{12}\sum_{\tau=-11}^{0}
        \sum_{i=1}^{N}\!\left(
        \frac{\texttt{saleq}_{i,t+\tau}}
             {\sum_{j=1}^{N}\texttt{saleq}_{j,t+\tau}}
        \right)^{\!2}

    where *N* is the number of firms in the firm's FF-49 industry group.
    The HHI is computed over the **full Compustat/CRSP universe** so
    that the measure is meaningful even when only a few tickers are
    requested.

    Quarterly-only.

    Reference: Hou & Robinson (2006).
    """
    from .industry_chars import _map_sic_to_industry  # local to avoid circular

    panel = raw_tables["__panel__"]
    engine = raw_tables.get("__engine__")

    # ── Fallback: panel-only HHI ────────────────────────────────────────
    def _panel_hhi():
        saleq = pd.to_numeric(panel["saleq"], errors="coerce")
        ind = panel["ind49"].astype(str)
        df = pd.DataFrame({"date": panel["date"], "permco": panel["permco"],
                            "ind49": ind, "saleq": saleq})
        df = df.loc[df["saleq"].notna() & (df["saleq"] > 0)]
        if df.empty:
            return pd.Series(np.nan, index=panel.index)
        ind_total = df.groupby(["date", "ind49"])["saleq"].transform("sum")
        df["share_sq"] = (df["saleq"] / ind_total) ** 2
        hhi = df.groupby(["date", "ind49"])["share_sq"].sum().reset_index()
        hhi.rename(columns={"share_sq": "hhi"}, inplace=True)
        firm_ind = pd.DataFrame({"date": panel["date"], "permco": panel["permco"],
                                  "ind49": ind})
        firm_ind = firm_ind.merge(hhi, on=["date", "ind49"], how="left")
        firm_ind["hhi"] = firm_ind["hhi"].astype(float)
        result = firm_ind.groupby("permco")["hhi"].transform(
            lambda x: x.rolling(window=12, min_periods=4).mean())
        return _quarter_end_only(
            pd.Series(result.values, index=panel.index), panel, freq)

    if engine is None or engine.wrds_conn is None:
        logger.warning("indcon: no engine — falling back to panel-only HHI")
        return _panel_hhi()

    # ── Full-universe sales + industry query ────────────────────────────
    dates = pd.to_datetime(panel["date"])
    start_str = dates.min().strftime("%Y-%m-%d")
    end_str   = dates.max().strftime("%Y-%m-%d")

    sf_table = "crsp.msf" if freq == "M" else "crsp.dsf"
    se_table = "crsp.mseall" if freq == "M" else "crsp.dseall"

    # Quarterly saleq per permco from Compustat via CCM link
    sql = (
        f"SELECT c.datadate AS date, l.lpermco AS permco, "
        f"c.saleq, b.hsiccd "
        f"FROM comp.fundq c "
        f"INNER JOIN crsp.ccmxpf_lnkhist l "
        f"ON c.gvkey = l.gvkey "
        f"AND l.linktype IN ('LU', 'LC') "
        f"AND l.linkprim IN ('P', 'C') "
        f"INNER JOIN {se_table} b "
        f"ON l.lpermco = b.permco "
        f"AND c.datadate = b.date "
        f"WHERE c.datadate BETWEEN '{start_str}' AND '{end_str}' "
        f"AND b.exchcd IN (1, 2, 3) "
        f"AND b.shrcd IN (10, 11) "
        f"AND c.saleq IS NOT NULL "
        f"AND c.saleq > 0"
    )
    logger.debug("indcon universe SQL: %s", sql)
    try:
        univ = engine.wrds_conn.raw_sql(sql)
    except Exception:
        logger.error("indcon: universe query failed — falling back",
                     exc_info=True)
        return _panel_hhi()

    if univ.empty:
        return pd.Series(np.nan, index=panel.index)

    univ["date"] = pd.to_datetime(univ["date"])
    if freq == "M":
        univ["date"] = univ["date"] + pd.offsets.MonthEnd(0)
    univ["saleq"] = pd.to_numeric(univ["saleq"], errors="coerce")
    univ["ind49"] = _map_sic_to_industry(univ["hsiccd"], 49)

    # Drop rows with missing data
    univ = univ.dropna(subset=["saleq", "ind49"])
    univ = univ[univ["saleq"] > 0]

    # Compute HHI per (date, ind49)
    ind_total = univ.groupby(["date", "ind49"])["saleq"].transform("sum")
    univ["share_sq"] = (univ["saleq"] / ind_total) ** 2
    hhi = (
        univ.groupby(["date", "ind49"])["share_sq"]
        .sum()
        .reset_index()
        .rename(columns={"share_sq": "hhi"})
    )

    # Merge HHI onto panel by (date, ind49)
    panel_tmp = panel[["date", "permco"]].copy()
    panel_tmp["ind49"] = panel["ind49"].astype(str)
    panel_tmp["date"] = pd.to_datetime(panel_tmp["date"])
    if freq == "M":
        panel_tmp["date"] = panel_tmp["date"] + pd.offsets.MonthEnd(0)
    merged = panel_tmp.merge(hhi, on=["date", "ind49"], how="left")

    # Rolling 4-quarter (12-month) mean of HHI per firm
    merged["hhi"] = merged["hhi"].astype(float)
    result = (
        merged.groupby("permco")["hhi"]
        .transform(lambda x: x.rolling(window=12, min_periods=4).mean())
    )
    return _quarter_end_only(
        pd.Series(result.values, index=panel.index), panel, freq,
    )

indcon.needs = {
    "comp.fundq": ["saleq"],
    "crsp.seall": ["hsiccd"],
}
indcon._output_name = "indcon"
indcon._order = 81
indcon._requires = ["ind49"]


# ═══════════════════════════════════════════════════════════════════════════
# hire  (order 82)  —  Belo, Lin & Bazdresch (2014)
# ═══════════════════════════════════════════════════════════════════════════

def hire(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Employee growth rate (annual).

    .. math::
        \texttt{hire}_t = \frac{\texttt{emp}_t - \texttt{emp}_{t-1}}
                               {\texttt{emp}_{t-1}}

    Uses annual Compustat (``comp.funda``), so the value is constant
    within each fiscal year and forward-filled by the engine.

    Reference: Belo, Lin & Bazdresch (2014).
    """
    panel = raw_tables["__panel__"]
    emp = pd.to_numeric(panel["emp"], errors="coerce")
    emp_lag = emp.groupby(panel["permco"]).shift(12)
    result = np.where(
        (emp_lag != 0) & emp_lag.notna() & emp.notna(),
        (emp - emp_lag) / emp_lag.abs(),
        np.nan,
    )
    return pd.Series(result, index=panel.index)

hire.needs = {"comp.funda": ["emp"]}
hire._output_name = "hire"
hire._order = 82


# ═══════════════════════════════════════════════════════════════════════════
# nincr  (order 83)  —  Barth, Elliott & Finn (1999)
# ═══════════════════════════════════════════════════════════════════════════

def nincr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Number of consecutive quarterly earnings increases (up to 8).

    .. math::
        \texttt{nincr}_t = \sum_{j=0}^{7}\;\prod_{\tau=0}^{j}
        \mathbf{1}\!\bigl\{\texttt{earn}_{t-\tau}
                           > \texttt{earn}_{t-\tau-1}\bigr\}

    Quarterly-only.

    Reference: Barth, Elliott & Finn (1999).
    """
    panel = raw_tables["__panel__"]
    earn = pd.to_numeric(panel["niq"], errors="coerce")

    # Build indicator: earn_t > earn_{t-1}  (quarterly lag = shift(3))
    indicators = []
    for tau in range(8):
        curr = earn.groupby(panel["permco"]).shift(3 * tau)
        prev = earn.groupby(panel["permco"]).shift(3 * (tau + 1))
        indicators.append((curr > prev).astype(float))

    # nincr = sum_{j=0}^{7} prod_{tau=0}^{j} indicator[tau]
    total = pd.Series(0.0, index=panel.index)
    running_prod = pd.Series(1.0, index=panel.index)
    for j in range(8):
        running_prod = running_prod * indicators[j]
        total = total + running_prod

    return _quarter_end_only(total, panel, freq)

nincr.needs = {"comp.fundq": ["niq"]}
nincr._output_name = "nincr"
nincr._order = 83
nincr._requires = ["earn"]


# ═══════════════════════════════════════════════════════════════════════════
# ps  (order 84)  —  Piotroski (2000)
# ═══════════════════════════════════════════════════════════════════════════

def ps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Piotroski F-Score (performance score).

    .. math::
        \texttt{ps}_t = \sum_{j=1}^{9} F_{j,t},
        \quad F_{j,t}\in\{0,1\}

    Nine binary indicators based on profitability, leverage, liquidity,
    and operating efficiency.

    Quarterly-only.

    Reference: Piotroski (2000).
    """
    panel = raw_tables["__panel__"]
    g = panel["permco"]

    niq    = pd.to_numeric(panel["niq"],     errors="coerce")
    oancfy = pd.to_numeric(panel["oancfy"],  errors="coerce")
    atq    = pd.to_numeric(panel["atq"],     errors="coerce")
    dlttq  = pd.to_numeric(panel["dlttq"],   errors="coerce")
    actq   = pd.to_numeric(panel["actq"],    errors="coerce")
    lctq   = pd.to_numeric(panel["lctq"],    errors="coerce")
    saleq  = pd.to_numeric(panel["saleq"],   errors="coerce")
    cogsq  = pd.to_numeric(panel["cogsq"],   errors="coerce")
    scstkcy = pd.to_numeric(panel["scstkcy"], errors="coerce")

    def _ttm(x):
        """Trailing-twelve-month (4-quarter) sum."""
        return x + x.groupby(g).shift(3) + x.groupby(g).shift(6) + x.groupby(g).shift(9)

    # Trailing 4Q sums — current window
    earn_ttm  = _ttm(niq)
    sale_ttm  = _ttm(saleq)
    cogs_ttm  = _ttm(cogsq)

    # Trailing 4Q sums — lagged window (t-4, i.e. shift(12))
    earn_ttm_lag  = earn_ttm.groupby(g).shift(12)
    sale_ttm_lag  = sale_ttm.groupby(g).shift(12)
    cogs_ttm_lag  = cogs_ttm.groupby(g).shift(12)

    atq_lag = atq.groupby(g).shift(12)

    # F1: trailing earnings > 0
    f1 = (earn_ttm > 0).astype(float)

    # F2: operating cash flow > 0
    f2 = (oancfy > 0).astype(float)

    # F3: ROA improvement
    roa_curr = np.where((atq != 0) & atq.notna(), earn_ttm / atq, np.nan)
    roa_lag  = np.where((atq_lag != 0) & atq_lag.notna(), earn_ttm_lag / atq_lag, np.nan)
    f3 = (pd.Series(roa_curr, index=panel.index) > pd.Series(roa_lag, index=panel.index)).astype(float)

    # F4: cash flow > earnings
    f4 = (oancfy > earn_ttm).astype(float)

    # F5: leverage decrease (dlttq/atq decreased)
    dlttq_lag = dlttq.groupby(g).shift(12)
    lev_curr = np.where((atq != 0) & atq.notna(), dlttq / atq, np.nan)
    lev_lag  = np.where((atq_lag != 0) & atq_lag.notna(), dlttq_lag / atq_lag, np.nan)
    f5 = (pd.Series(lev_lag, index=panel.index) > pd.Series(lev_curr, index=panel.index)).astype(float)

    # F6: liquidity improvement (current ratio increased)
    actq_lag = actq.groupby(g).shift(12)
    lctq_lag = lctq.groupby(g).shift(12)
    cr_curr = np.where((lctq != 0) & lctq.notna(), actq / lctq, np.nan)
    cr_lag  = np.where((lctq_lag != 0) & lctq_lag.notna(), actq_lag / lctq_lag, np.nan)
    f6 = (pd.Series(cr_curr, index=panel.index) > pd.Series(cr_lag, index=panel.index)).astype(float)

    # F7: gross margin improvement  (sale - cogs/sale)  per spec
    gm_curr = np.where(
        (sale_ttm != 0) & sale_ttm.notna(),
        sale_ttm - cogs_ttm / sale_ttm,
        np.nan,
    )
    gm_lag = np.where(
        (sale_ttm_lag != 0) & sale_ttm_lag.notna(),
        sale_ttm_lag - cogs_ttm_lag / sale_ttm_lag,
        np.nan,
    )
    f7 = (pd.Series(gm_curr, index=panel.index) > pd.Series(gm_lag, index=panel.index)).astype(float)

    # F8: asset turnover improvement (sale/atq)
    at_curr = np.where((atq != 0) & atq.notna(), sale_ttm / atq, np.nan)
    at_lag  = np.where((atq_lag != 0) & atq_lag.notna(), sale_ttm_lag / atq_lag, np.nan)
    f8 = (pd.Series(at_curr, index=panel.index) > pd.Series(at_lag, index=panel.index)).astype(float)

    # F9: no equity issuance
    f9 = (scstkcy.fillna(0) == 0).astype(float)

    score = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    return _quarter_end_only(score, panel, freq)

ps.needs = {
    "comp.fundq": ["niq", "oancfy", "atq", "dlttq", "actq",
                    "lctq", "saleq", "cogsq", "scstkcy"],
}
ps._output_name = "ps"
ps._order = 84


# ═══════════════════════════════════════════════════════════════════════════
# rsupq  (order 85)  —  Quarterly Revenue Surprise
# ═══════════════════════════════════════════════════════════════════════════

def rsupq(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarterly revenue surprise.

    .. math::
        \texttt{rsupq}_t = \frac{\texttt{saleq}_t - \texttt{saleq}_{t-1}}
                                 {\texttt{me}_t}

    Quarterly-only.
    """
    panel = raw_tables["__panel__"]
    saleq = pd.to_numeric(panel["saleq"], errors="coerce")
    saleq_lag = saleq.groupby(panel["permco"]).shift(3)
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & saleq.notna() & saleq_lag.notna(),
        (saleq - saleq_lag) / mkt.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

rsupq.needs = {
    "comp.fundq": ["saleq"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
rsupq._output_name = "rsupq"
rsupq._order = 85
rsupq._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# rsup  (order 86)  —  Kama (2009)
# ═══════════════════════════════════════════════════════════════════════════

def rsup(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Annual revenue surprise.

    .. math::
        \texttt{rsup}_t = \frac{\texttt{saleq}_t - \texttt{saleq}_{t-4}}
                                {\texttt{me}_t}

    Quarterly-only.

    Reference: Kama (2009).
    """
    panel = raw_tables["__panel__"]
    saleq = pd.to_numeric(panel["saleq"], errors="coerce")
    saleq_lag = saleq.groupby(panel["permco"]).shift(12)
    mkt = panel["me"].astype(float)
    result = np.where(
        (mkt != 0) & mkt.notna() & saleq.notna() & saleq_lag.notna(),
        (saleq - saleq_lag) / mkt.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

rsup.needs = {
    "comp.fundq": ["saleq"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
}
rsup._output_name = "rsup"
rsup._order = 86
rsup._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# debtequity  (order 87)  —  Debt-to-Equity
# ═══════════════════════════════════════════════════════════════════════════

def debtequity(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Debt-to-equity ratio.

    .. math::
        \texttt{debtequity}_t = \frac{\texttt{lctq}_t}{\texttt{ceqq}_t}

    Quarterly-only.
    """
    panel = raw_tables["__panel__"]
    lctq = pd.to_numeric(panel["lctq"], errors="coerce")
    ceqq = pd.to_numeric(panel["ceqq"], errors="coerce")
    result = np.where(
        (ceqq != 0) & ceqq.notna() & lctq.notna(),
        lctq / ceqq,
        np.nan,
    )
    return _quarter_end_only(pd.Series(result, index=panel.index), panel, freq)

debtequity.needs = {"comp.fundq": ["lctq", "ceqq"]}
debtequity._output_name = "debtequity"
debtequity._order = 87


# ═══════════════════════════════════════════════════════════════════════════
# earnvar  (order 88)  —  Earnings Variability
# ═══════════════════════════════════════════════════════════════════════════

def earnvar(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Earnings variability — rolling 20-quarter standard deviation of
    earnings.

    .. math::
        \texttt{earnvar}_t = \sigma(\texttt{earn}_{t-19},\ldots,
                                     \texttt{earn}_t)

    Only computed at calendar quarter-end months (March, June, September,
    December).  Uses the ``earn`` characteristic (= ``niq``) which is
    non-NaN only at quarter ends.

    Quarterly-only.
    """
    panel = raw_tables["__panel__"]
    earn = pd.to_numeric(panel["niq"], errors="coerce")

    # earn only has values at quarter-end months; compute rolling std
    # over the raw quarterly series (shift by 3 = one quarter)
    # We gather the last 20 quarterly observations per permco.
    date = pd.to_datetime(panel["date"])
    is_qe = date.dt.month.isin([3, 6, 9, 12])

    # Extract quarter-end rows, compute rolling std, merge back
    qe_mask = is_qe & earn.notna()
    qe_idx = panel.index[qe_mask]
    qe_earn = earn.loc[qe_idx]
    qe_permco = panel.loc[qe_idx, "permco"]

    qe_std = qe_earn.groupby(qe_permco).transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )

    result = pd.Series(np.nan, index=panel.index)
    result.loc[qe_idx] = qe_std.values

    return _quarter_end_only(result, panel, freq)

earnvar.needs = {"comp.fundq": ["niq"]}
earnvar._output_name = "earnvar"
earnvar._order = 88
earnvar._requires = ["earn"]


# ═══════════════════════════════════════════════════════════════════════════
# qual  (order 89)  —  MSCI Quality Index  —  Lettau & Ludvigson (2018)
# ═══════════════════════════════════════════════════════════════════════════

def qual(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""MSCI-style quality score.

    Cross-sectionally standardises ROE, debt-to-equity, and earnings
    variability across the **full Compustat/CRSP universe**, then
    combines them:

    .. math::
        \bar Z_t = \tfrac{1}{3}\bigl(Z^{\text{roe}}_t
                   + Z^{\text{de}}_t + Z^{\text{ev}}_t\bigr)

    .. math::
        \texttt{qual}_t = \begin{cases}
            1 + \bar Z_t & \text{if } \bar Z_t \ge 0 \\
            (1 - \bar Z_t)^{-1} & \text{otherwise}
        \end{cases}

    Quarterly-only.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    engine = raw_tables.get("__engine__")

    # ── Helper: panel-only z-score (fallback) ───────────────────────────
    def _panel_qual():
        roe_s = pd.to_numeric(panel.get("roe"), errors="coerce")
        de_s  = pd.to_numeric(panel.get("debtequity"), errors="coerce")
        ev_s  = pd.to_numeric(panel.get("earnvar"), errors="coerce")
        date_g = pd.to_datetime(panel["date"])

        def _zscore(s):
            mu = s.groupby(date_g).transform("mean")
            sd = s.groupby(date_g).transform("std")
            return np.where((sd != 0) & sd.notna(), (s - mu) / sd, np.nan)

        z_roe = pd.Series(_zscore(roe_s), index=panel.index)
        z_de  = pd.Series(_zscore(de_s),  index=panel.index)
        z_ev  = pd.Series(_zscore(ev_s),  index=panel.index)
        z_bar = (z_roe + z_de + z_ev) / 3.0
        res = np.where(z_bar >= 0, 1.0 + z_bar, 1.0 / (1.0 - z_bar))
        any_nan = z_roe.isna() | z_de.isna() | z_ev.isna()
        res = np.where(any_nan, np.nan, res)
        return _quarter_end_only(pd.Series(res, index=panel.index), panel, freq)

    if engine is None or engine.wrds_conn is None:
        logger.warning("qual: no engine — falling back to panel-only z-scores")
        return _panel_qual()

    # ── Full-universe query for ROE, debt-to-equity, earnings var ───────
    dates = pd.to_datetime(panel["date"])
    start_str = dates.min().strftime("%Y-%m-%d")
    end_str   = dates.max().strftime("%Y-%m-%d")

    se_table = "crsp.mseall" if freq == "M" else "crsp.dseall"

    # Query: ibq, ceqq, lctq, niq per (permco, datadate)
    sql = (
        f"SELECT c.datadate, l.lpermco AS permco, "
        f"c.ibq, c.ceqq, c.lctq, c.niq "
        f"FROM comp.fundq c "
        f"INNER JOIN crsp.ccmxpf_lnkhist l "
        f"ON c.gvkey = l.gvkey "
        f"AND l.linktype IN ('LU', 'LC') "
        f"AND l.linkprim IN ('P', 'C') "
        f"INNER JOIN {se_table} b "
        f"ON l.lpermco = b.permco "
        f"AND c.datadate = b.date "
        f"WHERE c.datadate BETWEEN '{start_str}' AND '{end_str}' "
        f"AND b.exchcd IN (1, 2, 3) "
        f"AND b.shrcd IN (10, 11)"
    )
    logger.debug("qual universe SQL: %s", sql)
    try:
        univ = engine.wrds_conn.raw_sql(sql)
    except Exception:
        logger.error("qual: universe query failed — falling back",
                     exc_info=True)
        return _panel_qual()

    if univ.empty:
        return pd.Series(np.nan, index=panel.index)

    univ["datadate"] = pd.to_datetime(univ["datadate"])
    if freq == "M":
        univ["datadate"] = univ["datadate"] + pd.offsets.MonthEnd(0)
    univ["permco"] = univ["permco"].astype("Int64")

    for c in ["ibq", "ceqq", "lctq", "niq"]:
        univ[c] = pd.to_numeric(univ[c], errors="coerce")

    # Compute ROE = ibq / lag(ceqq)  (lag within universe per permco)
    univ = univ.sort_values(["permco", "datadate"])
    univ["ceqq_lag"] = univ.groupby("permco")["ceqq"].shift(1)
    univ["roe_u"] = np.where(
        (univ["ceqq_lag"] != 0) & univ["ceqq_lag"].notna() & univ["ibq"].notna(),
        univ["ibq"] / univ["ceqq_lag"].abs(), np.nan)

    # Compute debt-to-equity = lctq / ceqq
    univ["de_u"] = np.where(
        (univ["ceqq"] != 0) & univ["ceqq"].notna() & univ["lctq"].notna(),
        univ["lctq"] / univ["ceqq"], np.nan)

    # Compute earnings variability = rolling 20-quarter std of niq
    univ["ev_u"] = (
        univ.groupby("permco")["niq"]
        .transform(lambda x: x.rolling(window=20, min_periods=5).std())
    )

    # Cross-sectional mean and std per date for z-scoring
    for var in ["roe_u", "de_u", "ev_u"]:
        mu = univ.groupby("datadate")[var].transform("mean")
        sd = univ.groupby("datadate")[var].transform("std")
        univ[f"z_{var}"] = np.where(
            (sd != 0) & sd.notna(), (univ[var] - mu) / sd, np.nan)

    univ["z_bar"] = (univ["z_roe_u"] + univ["z_de_u"] + univ["z_ev_u"]) / 3.0
    univ["qual_u"] = np.where(
        univ["z_bar"] >= 0,
        1.0 + univ["z_bar"],
        1.0 / (1.0 - univ["z_bar"]),
    )
    any_nan = (
        pd.Series(univ["z_roe_u"].values).isna()
        | pd.Series(univ["z_de_u"].values).isna()
        | pd.Series(univ["z_ev_u"].values).isna()
    )
    univ["qual_u"] = np.where(any_nan.values, np.nan, univ["qual_u"].values)

    # Keep only the needed columns and merge onto the panel
    qual_df = univ[["datadate", "permco", "qual_u"]].copy()
    qual_df.rename(columns={"datadate": "date"}, inplace=True)

    panel_tmp = panel[["date", "permco"]].copy()
    panel_tmp["date"] = pd.to_datetime(panel_tmp["date"])
    if freq == "M":
        panel_tmp["date"] = panel_tmp["date"] + pd.offsets.MonthEnd(0)
    panel_tmp["permco"] = panel_tmp["permco"].astype("Int64")

    merged = panel_tmp.merge(qual_df, on=["date", "permco"], how="left")

    return _quarter_end_only(
        pd.Series(merged["qual_u"].values, index=panel.index), panel, freq,
    )

qual.needs = {
    "comp.fundq": ["ibq", "ceqq", "lctq", "niq"],
}
qual._output_name = "qual"
qual._order = 89
qual._requires = ["roe", "debtequity", "earnvar"]


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


# ═══════════════════════════════════════════════════════════════════════════
# eps  (order 90)  —  Earnings per Share
# ═══════════════════════════════════════════════════════════════════════════

def eps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Earnings per share (trailing four-quarter sum).

    .. math:: \texttt{eps}_t
              = \frac{\sum_{\tau=-3}^{0} \texttt{earn}_{t+\tau}}
                     {\texttt{shrout}_t}
    """
    panel = raw_tables["__panel__"]
    earn_filled = panel.groupby("permco")["earn"].ffill().astype(float)
    earn_4q = (
        earn_filled
        + earn_filled.groupby(panel["permco"]).shift(3)
        + earn_filled.groupby(panel["permco"]).shift(6)
        + earn_filled.groupby(panel["permco"]).shift(9)
    )
    shrout_vals = panel["shrout"].astype(float)
    result = np.where(
        (shrout_vals != 0) & shrout_vals.notna() & earn_4q.notna(),
        earn_4q / shrout_vals,
        np.nan,
    )
    return pd.Series(result, index=panel.index, dtype=float)

eps.needs = {
    "comp.fundq": ["niq"],
    "crsp.sf": ["shrout", "cfacshr"],
}
eps._output_name = "eps"
eps._order = 90
eps._requires = ["earn", "shrout"]


# ═══════════════════════════════════════════════════════════════════════════
# eps_gr  (order 91)  —  Earnings per Share Growth (QoQ)
# ═══════════════════════════════════════════════════════════════════════════

def eps_gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Quarter-over-quarter earnings-per-share growth.

    .. math:: \texttt{eps\_gr}_t
              = \frac{\texttt{eps}_t - \texttt{eps}_{t-1}}{|\texttt{eps}_{t-1}|}

    Quarterly-only: populated only at calendar quarter-end months.
    """
    panel = raw_tables["__panel__"]
    vals = panel["eps"].astype(float)
    lag = panel.groupby("permco")["eps"].shift(3).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

eps_gr.needs = {
    "comp.fundq": ["niq"],
    "crsp.sf": ["shrout", "cfacshr"],
}
eps_gr._output_name = "eps_gr"
eps_gr._order = 91
eps_gr._requires = ["eps"]


# ═══════════════════════════════════════════════════════════════════════════
# eps_gr_yoy  (order 92)  —  YoY Earnings per Share Growth
# ═══════════════════════════════════════════════════════════════════════════

def eps_gr_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year earnings-per-share growth.

    .. math:: \texttt{eps\_gr\_yoy}_t
              = \frac{\texttt{eps}_t - \texttt{eps}_{t-4}}{|\texttt{eps}_{t-4}|}

    Quarterly-only: populated only at calendar quarter-end months.
    """
    panel = raw_tables["__panel__"]
    vals = panel["eps"].astype(float)
    lag = panel.groupby("permco")["eps"].shift(12).astype(float)
    growth = np.where(
        (lag != 0) & lag.notna() & vals.notna(),
        (vals - lag) / lag.abs(),
        np.nan,
    )
    return _quarter_end_only(pd.Series(growth, index=panel.index), panel, freq)

eps_gr_yoy.needs = {
    "comp.fundq": ["niq"],
    "crsp.sf": ["shrout", "cfacshr"],
}
eps_gr_yoy._output_name = "eps_gr_yoy"
eps_gr_yoy._order = 92
eps_gr_yoy._requires = ["eps"]


# ═══════════════════════════════════════════════════════════════════════════
# sue  (order 93)  —  Standardized Unexpected Earnings  (Rendleman 1982)
# ═══════════════════════════════════════════════════════════════════════════

def sue(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Standardized unexpected earnings.

    .. math::
        \texttt{sue}_t = \frac{\texttt{eps}_t - \hat{\texttt{eps}}_t}
                              {\sigma(\hat{\varepsilon})}

    where :math:`\hat{\texttt{eps}}_t` comes from a rolling 20-quarter OLS:

    .. math::
        \texttt{eps}_q = \theta_0 + \theta_1 t + \theta_2 t^2
                       + \theta_3 \delta_{Q2} + \theta_4 \delta_{Q3}
                       + \theta_5 \delta_{Q4} + \varepsilon

    Reference: Rendleman, Jones & Latané (1982).
    """
    panel = raw_tables["__panel__"]
    if freq != "M":
        return pd.Series(np.nan, index=panel.index)

    panel_date = pd.to_datetime(panel["date"])
    is_qe = panel_date.dt.month.isin([3, 6, 9, 12])

    eps_vals = panel["eps"].astype(float)
    result = pd.Series(np.nan, index=panel.index, dtype=float)

    min_obs = 20  # require 20 quarters of history

    for pc, grp in panel.loc[is_qe].groupby("permco"):
        y = grp["eps"].astype(float).values
        n = len(y)
        if n < min_obs:
            continue

        # Build regressors: constant, t, t^2, Q2/Q3/Q4 dummies
        months = pd.to_datetime(grp["date"]).dt.month.values
        t_idx = np.arange(n, dtype=float)

        X = np.column_stack([
            np.ones(n),
            t_idx,
            t_idx ** 2,
            (months == 6).astype(float),   # δ_Q2
            (months == 9).astype(float),   # δ_Q3
            (months == 12).astype(float),  # δ_Q4
        ])

        idx_list = grp.index.tolist()

        for i in range(min_obs, n):
            y_win = y[i - min_obs : i + 1]
            X_win = X[i - min_obs : i + 1]

            if np.isnan(y_win).any():
                continue

            # Fit on first min_obs obs, predict the last one
            y_fit = y_win[:-1]
            X_fit = X_win[:-1]
            x_pred = X_win[-1:]

            try:
                beta, residuals, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
            except np.linalg.LinAlgError:
                continue

            y_hat = x_pred @ beta
            resid = y_fit - X_fit @ beta
            sigma = resid.std(ddof=1)

            if sigma > 0:
                result.loc[idx_list[i]] = (y_win[-1] - y_hat[0]) / sigma

    return result

sue.needs = {
    "comp.fundq": ["niq"],
    "crsp.sf": ["shrout", "cfacshr"],
}
sue._output_name = "sue"
sue._order = 93
sue._requires = ["eps"]


# ═══════════════════════════════════════════════════════════════════════════
# mult  (order 98)  —  Multiples Index  (Lettau & Ludvigson 2018)
# ═══════════════════════════════════════════════════════════════════════════

# ── Module-level cache for full-universe characteristic data ─────────────
_UNIV_CACHE: dict[tuple, pd.DataFrame | None] = {}


def _build_universe_chars(
    engine,
    freq: str,
    start_str: str,
    end_str: str,
) -> pd.DataFrame | None:
    """Compute characteristic values for the full CRSP default universe.

    Queries CRSP, Compustat FUNDQ, the CCM link table, and IBES from
    WRDS, then replicates the same variable definitions used by the
    panel characteristic functions.  The result is cached by
    ``(freq, start_str, end_str)`` so that ``mult`` and ``gr``
    share a single WRDS round-trip.

    Returns
    -------
    pd.DataFrame | None
        Columns: ``date, permco, bm, sp, cfp, dy, ep1,
        earn_gr_yoy, s_gr_yoy, cf_gr_yoy, be_gr_yoy, eltg``.
        ``None`` if any query fails.
    """
    cache_key = (freq, start_str, end_str)
    if cache_key in _UNIV_CACHE:
        return _UNIV_CACHE[cache_key]

    if engine is None or engine.wrds_conn is None:
        _UNIV_CACHE[cache_key] = None
        return None

    sf_table = "crsp.msf" if freq == "M" else "crsp.dsf"
    se_table = "crsp.mseall" if freq == "M" else "crsp.dseall"

    # Extend start dates so trailing sums and lags are valid at start_str
    pad_crsp = pd.DateOffset(months=13)   # dy needs 12-month trailing
    pad_comp = pd.DateOffset(months=18)   # rolling 4Q + YoY on quarterly
    crsp_start = (pd.Timestamp(start_str) - pad_crsp).strftime("%Y-%m-%d")
    comp_start = (pd.Timestamp(start_str) - pad_comp).strftime("%Y-%m-%d")

    # ── 1. CRSP monthly stock file ───────────────────────────────────────
    crsp_sql = (
        f"SELECT a.date, a.permco, "
        f"ABS(a.prc) / NULLIF(a.cfacpr, 0) AS adj_prc, "
        f"a.shrout * a.cfacshr AS adj_shrout, "
        f"a.ret, a.retx, b.cusip "
        f"FROM {sf_table} a "
        f"INNER JOIN {se_table} b "
        f"ON a.permco = b.permco AND a.date = b.date "
        f"WHERE a.date BETWEEN '{crsp_start}' AND '{end_str}' "
        f"AND b.exchcd IN (1, 2, 3) "
        f"AND b.shrcd IN (10, 11) "
        f"AND a.prc IS NOT NULL "
        f"AND a.shrout IS NOT NULL "
        f"AND a.cfacpr IS NOT NULL AND a.cfacpr <> 0"
    )

    # ── 2. Compustat FUNDQ ───────────────────────────────────────────────
    comp_sql = (
        f"SELECT gvkey, datadate, seqq, txditcq, pstkrq, pstkq, "
        f"saleq, ibq, dpq, niq "
        f"FROM comp.fundq "
        f"WHERE datadate BETWEEN '{comp_start}' AND '{end_str}' "
        f"AND (seqq IS NOT NULL OR niq IS NOT NULL OR saleq IS NOT NULL)"
    )

    # ── 3. CCM link table ────────────────────────────────────────────────
    link_sql = (
        "SELECT gvkey, lpermco AS permco, linkdt, linkenddt "
        "FROM crsp.ccmxpf_lnkhist "
        "WHERE linktype IN ('LU', 'LC') "
        "AND linkprim IN ('P', 'C')"
    )

    # ── 4. IBES detail estimates (fpi 1 and 3) ──────────────────────────
    ibes_sql = (
        f"SELECT cusip, fpi, anndats, analys, value "
        f"FROM ibes.det_epsus "
        f"WHERE anndats BETWEEN '{start_str}' AND '{end_str}' "
        f"AND fpi IN ('1', '3') "
        f"AND value IS NOT NULL"
    )

    logger.info("_build_universe_chars: fetching full CRSP universe …")
    try:
        crsp = engine.wrds_conn.raw_sql(crsp_sql)
        comp = engine.wrds_conn.raw_sql(comp_sql)
        link = engine.wrds_conn.raw_sql(link_sql)
        ibes_raw = engine.wrds_conn.raw_sql(ibes_sql)
    except Exception:
        logger.error("_build_universe_chars: WRDS queries failed",
                     exc_info=True)
        _UNIV_CACHE[cache_key] = None
        return None

    if crsp.empty:
        logger.warning("_build_universe_chars: CRSP query returned empty")
        _UNIV_CACHE[cache_key] = None
        return None

    # ── Process CRSP ─────────────────────────────────────────────────────
    crsp["date"] = pd.to_datetime(crsp["date"])
    if freq == "M":
        crsp["date"] = crsp["date"] + pd.offsets.MonthEnd(0)
    crsp["permco"] = crsp["permco"].astype("Int64")
    for c in ("adj_prc", "adj_shrout", "ret", "retx"):
        crsp[c] = pd.to_numeric(crsp[c], errors="coerce")
    crsp["me"] = crsp["adj_prc"] * crsp["adj_shrout"]
    crsp["cusip"] = crsp["cusip"].astype(str).str.strip()

    # Dedup: one row per (permco, date) — keep largest ME share class
    crsp = crsp.sort_values(["permco", "date", "me"],
                            ascending=[True, True, False])
    crsp = crsp.drop_duplicates(subset=["permco", "date"], keep="first")

    # Dividend yield: trailing 12-month Σ[(ret-retx)*ME_{t-1}] / ME_t
    crsp = crsp.sort_values(["permco", "date"])
    crsp["me_lag"] = crsp.groupby("permco")["me"].shift(1)
    crsp["div_dollar"] = (crsp["ret"] - crsp["retx"]) * crsp["me_lag"]
    crsp["dy"] = (
        crsp.groupby("permco")["div_dollar"]
        .transform(lambda x: x.rolling(12, min_periods=1).sum())
    )
    crsp["dy"] = np.where(
        (crsp["me"] != 0) & crsp["me"].notna(),
        crsp["dy"] / crsp["me"],
        np.nan,
    )

    # ── Process Compustat ────────────────────────────────────────────────
    has_comp = not comp.empty
    if has_comp:
        comp["datadate"] = pd.to_datetime(comp["datadate"])
        for c in ("seqq", "txditcq", "pstkrq", "pstkq",
                   "saleq", "ibq", "dpq", "niq"):
            if c in comp.columns:
                comp[c] = pd.to_numeric(comp[c], errors="coerce")

        # Book equity
        ps = comp[["pstkrq", "pstkq"]].bfill(axis=1).iloc[:, 0].fillna(0)
        comp["be"] = (comp["seqq"].fillna(0)
                      + comp["txditcq"].fillna(0) - ps)
        # Cashflow = ibq + dpq (dpq → 0 if missing)
        comp["cf"] = comp["ibq"].fillna(0) + comp["dpq"].fillna(0)

        comp = comp.sort_values(["gvkey", "datadate"])

        # Trailing 4-quarter rolling sums (quarterly level)
        for src, dst in [("saleq", "sale4q"),
                         ("cf",    "cf4q"),
                         ("niq",   "ni4q")]:
            comp[dst] = comp.groupby("gvkey")[src].transform(
                lambda s: s.rolling(4, min_periods=4).sum()
            )

        # Year-over-year growth (shift 4 quarters)
        for src, dst in [("niq",   "earn_gr_yoy"),
                         ("saleq", "s_gr_yoy"),
                         ("cf",    "cf_gr_yoy")]:
            lag = comp.groupby("gvkey")[src].shift(4)
            comp[dst] = np.where(
                (lag != 0) & lag.notna() & comp[src].notna(),
                (comp[src] - lag) / lag.abs(),
                np.nan,
            )
        # be_gr_yoy = be / be_{t-4}  (ratio, not difference/denom)
        be_lag = comp.groupby("gvkey")["be"].shift(4)
        comp["be_gr_yoy"] = np.where(
            (be_lag > 0) & be_lag.notna() & comp["be"].notna(),
            comp["be"] / be_lag,
            np.nan,
        )

        # Merge CCM link → permco (dedup one permco per gvkey)
        link["permco"] = link["permco"].astype("Int64")
        comp = comp.merge(
            link[["gvkey", "permco"]].drop_duplicates(subset=["gvkey"]),
            on="gvkey", how="inner",
        )
        comp = comp.sort_values("datadate")

        # merge_asof: for each (permco, date) in CRSP, pick the most
        # recent Compustat quarter
        crsp = crsp.sort_values("date")
        merge_cols = [
            "permco", "datadate", "be", "sale4q", "cf4q", "ni4q",
            "earn_gr_yoy", "s_gr_yoy", "cf_gr_yoy", "be_gr_yoy",
        ]
        crsp = pd.merge_asof(
            crsp, comp[merge_cols],
            left_on="date", right_on="datadate",
            by="permco", direction="backward",
        )

        # Ratio characteristics
        pos_me = (crsp["me"] > 0) & crsp["me"].notna()
        crsp["bm"] = np.where(
            pos_me & crsp["be"].notna() & (crsp["be"] > 0),
            crsp["be"] / crsp["me"], np.nan,
        )
        crsp["sp"] = np.where(
            pos_me & crsp["sale4q"].notna(),
            crsp["sale4q"] / crsp["me"], np.nan,
        )
        crsp["cfp"] = np.where(
            pos_me & crsp["cf4q"].notna(),
            crsp["cf4q"] / crsp["me"], np.nan,
        )
        # Trailing 4Q EPS (for eltg)
        crsp["__eps_trail__"] = np.where(
            (crsp["adj_shrout"] > 0) & crsp["adj_shrout"].notna()
            & crsp["ni4q"].notna(),
            crsp["ni4q"] / crsp["adj_shrout"], np.nan,
        )
    else:
        for col in ("bm", "sp", "cfp", "earn_gr_yoy", "s_gr_yoy",
                     "cf_gr_yoy", "be_gr_yoy", "__eps_trail__"):
            crsp[col] = np.nan

    # ── Process IBES ─────────────────────────────────────────────────────
    if not ibes_raw.empty:
        ibes_raw["anndats"] = pd.to_datetime(
            ibes_raw["anndats"], errors="coerce"
        )
        ibes_raw["anndats"] = ibes_raw["anndats"] + pd.offsets.MonthEnd(0)
        ibes_raw = ibes_raw.dropna(subset=["anndats", "cusip"])
        ibes_raw["cusip"] = ibes_raw["cusip"].str.strip()
        ibes_raw["value"] = pd.to_numeric(ibes_raw["value"], errors="coerce")

        # Mean consensus by (date, cusip, fpi)
        consensus = (
            ibes_raw.groupby(["anndats", "cusip", "fpi"])["value"]
            .mean()
            .reset_index()
            .rename(columns={"anndats": "date"})
        )

        # fpi = 1 → ep1 = consensus EPS / price
        cons_1 = (
            consensus[consensus["fpi"].astype(str) == "1"]
            [["date", "cusip", "value"]]
            .copy()
            .rename(columns={"value": "__eps1__"})
        )
        crsp = crsp.merge(cons_1, on=["date", "cusip"], how="left")
        crsp["ep1"] = np.where(
            (crsp["adj_prc"] > 0) & crsp["adj_prc"].notna()
            & crsp["__eps1__"].notna(),
            crsp["__eps1__"] / crsp["adj_prc"], np.nan,
        )

        # fpi = 3 → eltg = consensus EPS3 − trailing 4Q EPS
        cons_3 = (
            consensus[consensus["fpi"].astype(str) == "3"]
            [["date", "cusip", "value"]]
            .copy()
            .rename(columns={"value": "__eps3__"})
        )
        crsp = crsp.merge(cons_3, on=["date", "cusip"], how="left")
        crsp["eltg"] = np.where(
            crsp["__eps3__"].notna() & crsp["__eps_trail__"].notna(),
            crsp["__eps3__"] - crsp["__eps_trail__"], np.nan,
        )
    else:
        crsp["ep1"] = np.nan
        crsp["eltg"] = np.nan

    # ── Trim to the requested date range ─────────────────────────────────
    crsp = crsp[crsp["date"] >= pd.Timestamp(start_str)].copy()

    out_cols = [
        "date", "permco", "bm", "sp", "cfp", "dy", "ep1",
        "earn_gr_yoy", "s_gr_yoy", "cf_gr_yoy", "be_gr_yoy",
        "eltg",
    ]
    result = crsp[[c for c in out_cols if c in crsp.columns]].copy()
    logger.info(
        "_build_universe_chars: %d rows, %d dates, %d permcos",
        len(result), result["date"].nunique(), result["permco"].nunique(),
    )
    _UNIV_CACHE[cache_key] = result
    return result


def _full_universe_rank(
    panel: pd.DataFrame,
    var_name: str,
    raw_tables: dict[str, pd.DataFrame],
    freq: str,
) -> pd.Series:
    """Cross-sectional percentile rank (0–100) against full CRSP universe.

    For each panel row, computes::

        rank = (# universe stocks below + 0.5 × # equal) / N × 100

    Falls back to panel-only rank when the universe data is unavailable
    or the panel already contains ≥ 30 stocks.
    """
    panel_var = (
        panel[var_name].astype(float)
        if var_name in panel.columns
        else pd.Series(np.nan, index=panel.index)
    )

    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        logger.warning("%s: no engine — falling back to panel-only rank",
                       var_name)
        return (panel.assign(__v=panel_var)
                .groupby("date")["__v"]
                .rank(pct=True) * 100)

    # If the panel already has many stocks, panel-only rank is fine
    n_stocks = panel.groupby("date")["permco"].nunique().median()
    if n_stocks >= 30:
        return (panel.assign(__v=panel_var)
                .groupby("date")["__v"]
                .rank(pct=True) * 100)

    dates = pd.to_datetime(panel["date"])
    start_str = dates.min().strftime("%Y-%m-%d")
    end_str = dates.max().strftime("%Y-%m-%d")

    univ = _build_universe_chars(engine, freq, start_str, end_str)

    if univ is None or var_name not in univ.columns:
        logger.warning(
            "%s: universe data unavailable — panel-only rank", var_name,
        )
        return (panel.assign(__v=panel_var)
                .groupby("date")["__v"]
                .rank(pct=True) * 100)

    # Pre-group universe values by date for efficient lookup
    univ_trimmed = univ[["date", var_name]].dropna(subset=[var_name])
    univ_by_date: dict = {
        dt: grp[var_name].values
        for dt, grp in univ_trimmed.groupby("date")
    }

    panel_dates = pd.to_datetime(panel["date"])
    if freq == "M":
        panel_dates = panel_dates + pd.offsets.MonthEnd(0)
    pvals = panel_var.values

    result = np.full(len(panel), np.nan, dtype=float)
    for i in range(len(panel)):
        pv = pvals[i]
        if pd.isna(pv):
            continue
        uv = univ_by_date.get(panel_dates.iloc[i])
        if uv is None or len(uv) == 0:
            continue
        n = len(uv)
        result[i] = (np.sum(uv < pv) + 0.5 * np.sum(uv == pv)) / n * 100

    return pd.Series(result, index=panel.index, dtype=float)


def mult(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Multiples index.

    .. math::
        \texttt{mult}_t = \tfrac{1}{2}\operatorname{pr}(\texttt{ep1}_t)
                        + \tfrac{1}{8}\bigl[\operatorname{pr}(\texttt{bm}_t)
                        + \operatorname{pr}(\texttt{sp}_t)
                        + \operatorname{pr}(\texttt{cfp}_t)
                        + \operatorname{pr}(\texttt{dy}_t)\bigr]

    where :math:`\operatorname{pr}(\cdot)` is the cross-sectional
    percentile rank (1–100) computed over the full CRSP default universe.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]

    pr_ep1 = _full_universe_rank(panel, "ep1", raw_tables, freq)
    pr_bm  = _full_universe_rank(panel, "bm",  raw_tables, freq)
    pr_sp  = _full_universe_rank(panel, "sp",  raw_tables, freq)
    pr_cfp = _full_universe_rank(panel, "cfp", raw_tables, freq)
    pr_dy  = _full_universe_rank(panel, "dy",  raw_tables, freq)

    result = 0.5 * pr_ep1 + 0.125 * (pr_bm + pr_sp + pr_cfp + pr_dy)
    return _quarter_end_only(result, panel, freq)

mult.needs = {
    "ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr", "ret", "retx"],
    "comp.fundq": ["seqq", "txditcq", "pstkrq", "pstkq", "saleq", "ibq", "dpq"],
}
mult._output_name = "mult"
mult._order = 98
mult._requires = ["ep1", "bm", "sp", "cfp", "dy"]


# ═══════════════════════════════════════════════════════════════════════════
# gr  (order 99)  —  Growth Index  (Lettau & Ludvigson 2018)
# ═══════════════════════════════════════════════════════════════════════════

def gr(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Growth index (expected LT earnings growth).

    .. math::
        \texttt{gr}_t = \tfrac{1}{2}\operatorname{pr}(\texttt{eltg}_t)
                       + \tfrac{1}{8}\bigl[\operatorname{pr}(\texttt{earn\_gr\_yoy}_t)
                       + \operatorname{pr}(\texttt{s\_gr\_yoy}_t)
                       + \operatorname{pr}(\texttt{cf\_gr\_yoy}_t)
                       + \operatorname{pr}(\texttt{be\_gr\_yoy}_t)\bigr]

    where :math:`\operatorname{pr}(\cdot)` is the cross-sectional
    percentile rank (1–100) computed over the full CRSP default universe.

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]

    pr_eltg    = _full_universe_rank(panel, "eltg",        raw_tables, freq)
    pr_earn_gr = _full_universe_rank(panel, "earn_gr_yoy", raw_tables, freq)
    pr_s_gr    = _full_universe_rank(panel, "s_gr_yoy",    raw_tables, freq)
    pr_cf_gr   = _full_universe_rank(panel, "cf_gr_yoy",   raw_tables, freq)
    pr_be_gr   = _full_universe_rank(panel, "be_gr_yoy",   raw_tables, freq)

    result = 0.5 * pr_eltg + 0.125 * (pr_earn_gr + pr_s_gr + pr_cf_gr + pr_be_gr)
    return _quarter_end_only(result, panel, freq)

gr.needs = {
    "ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"],
    "crsp.sf": ["shrout", "cfacshr"],
    "comp.fundq": ["niq", "saleq", "ibq", "dpq", "seqq", "txditcq", "pstkrq", "pstkq"],
}
gr._output_name = "gr"
gr._order = 99
gr._requires = ["eltg", "earn_gr_yoy", "s_gr_yoy", "cf_gr_yoy", "be_gr_yoy"]


# ═══════════════════════════════════════════════════════════════════════════
# ms  (order 101)  —  Morningstar Index  (Lettau & Ludvigson 2018)
# ═══════════════════════════════════════════════════════════════════════════

def ms(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Morningstar index.

    .. math:: \texttt{ms}_t = \texttt{mult}_t - \texttt{gr}_t

    Reference: Lettau & Ludvigson (2018).
    """
    panel = raw_tables["__panel__"]
    mult_vals = panel["mult"].astype(float) if "mult" in panel.columns else pd.Series(np.nan, index=panel.index)
    gr_vals   = panel["gr"].astype(float) if "gr" in panel.columns else pd.Series(np.nan, index=panel.index)
    result = mult_vals - gr_vals
    return _quarter_end_only(result, panel, freq)

ms.needs = {
    "ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"],
    "crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr", "ret", "retx"],
    "comp.fundq": ["niq", "seqq", "txditcq", "pstkrq", "pstkq", "saleq", "ibq", "dpq"],
}
ms._output_name = "ms"
ms._order = 101
ms._requires = ["mult", "gr"]



