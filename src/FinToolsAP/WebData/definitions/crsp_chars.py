"""
FinToolsAP.WebData.definitions.crsp_chars
==========================================

Built-in characteristics derived from CRSP stock files (MSF / DSF)
and CRSP security-event files (MSEALL / DSEALL).

Characteristics
---------------
prc          split-adjusted absolute price           (order  2)
shrout       split-adjusted shares outstanding       (order  3)
shrout_ch    month-over-month change in shrout       (order  4)
shrout_ch_yoy  year-over-year change in shrout       (order  5)
me           market equity (prc × shrout)            (order  6)
me_ch        month-over-month change in ME           (order  7)
me_ch_yoy    year-over-year change in ME             (order  8)
me_ia        industry-adjusted ME (FF-49)            (order  9)
bas          bid-ask spread                          (order 11)
bas_r3m      rolling 3-month bid-ask spread          (order 12)
beta_r3m     rolling 3-month CAPM beta               (order 13)
dvol         log dollar volume                       (order 14)
dy           dividend yield                          (order 15)
illiq        Amihud illiquidity                      (order 16)
rvar_capm    CAPM residual variance (3m)             (order 17)
rvar_ff3     FF3 residual variance (3m)              (order 18)
rvar_ff5     FF5 residual variance (3m)              (order 19)
rvar_car     Carhart residual variance (3m)          (order 20)
rvar_mean    return variance (63d)                   (order 21)
std_dvol     std of dollar volume (63d)              (order 22)
turn         share turnover                          (order 23)
std_turn     std of share turnover (63d)             (order 24)
zerotrade    zero-trading days (63d)                 (order 25)
psliq        Pastor-Stambaugh liquidity              (order 26)

Convention
----------
* Each function receives ``(raw_tables, freq)`` and returns a
  ``pandas.Series`` aligned to the panel index.
* Lag lengths adapt automatically to frequency (monthly vs daily).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: frequency-aware lag
# ═══════════════════════════════════════════════════════════════════════════

def _lag(freq: str, monthly_lag: int = 1) -> int:
    """Return the number of periods corresponding to *monthly_lag* months.

    For daily frequency the approximate trading-day equivalent is used
    (21 trading days ≈ 1 calendar month).
    """
    return monthly_lag if freq == "M" else monthly_lag * 21


# ═══════════════════════════════════════════════════════════════════════════
# prc  (order 2)
# ═══════════════════════════════════════════════════════════════════════════

def prc(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Split-adjusted absolute stock price.

    .. math:: \texttt{prc}_t = \frac{|\texttt{prc}_t|}{\texttt{cfacpr}_t}
    """
    panel = raw_tables["__panel__"]
    return panel["prc"].abs() / panel["cfacpr"]

prc.needs = {"crsp.sf": ["prc", "cfacpr"]}
prc._output_name = "prc"
prc._order = 2


# ═══════════════════════════════════════════════════════════════════════════
# shrout  (order 3)
# ═══════════════════════════════════════════════════════════════════════════

def shrout(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Split-adjusted shares outstanding.

    .. math:: \texttt{shrout}_t = \texttt{shrout}_t \times \texttt{cfacshr}_t
    """
    panel = raw_tables["__panel__"]
    return panel["shrout"] * panel["cfacshr"]

shrout.needs = {"crsp.sf": ["shrout", "cfacshr"]}
shrout._output_name = "shrout"
shrout._order = 3


# ═══════════════════════════════════════════════════════════════════════════
# shrout_ch  (order 4)
# ═══════════════════════════════════════════════════════════════════════════

def shrout_ch(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Month-over-month proportional change in shares outstanding.

    .. math:: \texttt{shrout\_ch}_t
              = \frac{\texttt{shrout}_t}{\texttt{shrout}_{t-1}} - 1
    """
    panel = raw_tables["__panel__"]
    lag = _lag(freq, 1)
    lagged = panel.groupby("permco")["shrout"].shift(lag)
    return panel["shrout"] / lagged - 1

shrout_ch.needs = {"crsp.sf": ["shrout", "cfacshr"]}
shrout_ch._output_name = "shrout_ch"
shrout_ch._order = 4
shrout_ch._requires = ["shrout"]


# ═══════════════════════════════════════════════════════════════════════════
# shrout_ch_yoy  (order 5)  —  Pontiff & Woodgate (2008)
# ═══════════════════════════════════════════════════════════════════════════

def shrout_ch_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year proportional change in shares outstanding.

    .. math:: \texttt{shrout\_ch\_yoy}_t
              = \frac{\texttt{shrout}_t}{\texttt{shrout}_{t-12}} - 1

    Reference: Pontiff & Woodgate (2008).
    """
    panel = raw_tables["__panel__"]
    lag = _lag(freq, 12)
    lagged = panel.groupby("permco")["shrout"].shift(lag)
    return panel["shrout"] / lagged - 1

shrout_ch_yoy.needs = {"crsp.sf": ["shrout", "cfacshr"]}
shrout_ch_yoy._output_name = "shrout_ch_yoy"
shrout_ch_yoy._order = 5
shrout_ch_yoy._requires = ["shrout"]


# ═══════════════════════════════════════════════════════════════════════════
# me  (order 6)  —  Banz (1981)
# ═══════════════════════════════════════════════════════════════════════════

def me(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Market equity.

    .. math:: \texttt{me}_t = \texttt{prc}_t \times \texttt{shrout}_t

    Both ``prc`` and ``shrout`` are already split-adjusted by their
    respective order-2 and order-3 definitions.

    Reference: Banz (1981).
    """
    panel = raw_tables["__panel__"]
    return panel["prc"] * panel["shrout"]

me.needs = {"crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"]}
me._output_name = "me"
me._order = 6
me._requires = ["prc", "shrout"]


# ═══════════════════════════════════════════════════════════════════════════
# me_ch  (order 7)
# ═══════════════════════════════════════════════════════════════════════════

def me_ch(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Month-over-month proportional change in market equity.

    .. math:: \texttt{me\_ch}_t
              = \frac{\texttt{me}_t}{\texttt{me}_{t-1}} - 1
    """
    panel = raw_tables["__panel__"]
    lag = _lag(freq, 1)
    lagged = panel.groupby("permco")["me"].shift(lag)
    return panel["me"] / lagged - 1

me_ch.needs = {"crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"]}
me_ch._output_name = "me_ch"
me_ch._order = 7
me_ch._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# me_ch_yoy  (order 8)
# ═══════════════════════════════════════════════════════════════════════════

def me_ch_yoy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Year-over-year proportional change in market equity.

    .. math:: \texttt{me\_ch\_yoy}_t
              = \frac{\texttt{me}_t}{\texttt{me}_{t-12}} - 1
    """
    panel = raw_tables["__panel__"]
    lag = _lag(freq, 12)
    lagged = panel.groupby("permco")["me"].shift(lag)
    return panel["me"] / lagged - 1

me_ch_yoy.needs = {"crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"]}
me_ch_yoy._output_name = "me_ch_yoy"
me_ch_yoy._order = 8
me_ch_yoy._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# me_ia  (order 9)  —  Asness et al. (2000)
# ═══════════════════════════════════════════════════════════════════════════

def me_ia(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Industry-adjusted market equity (FF-49).

    .. math:: \texttt{me\_ia}_t
              = \texttt{me}_t
              - \overline{\texttt{me}_t}^{\,\text{FF-49}}

    Subtracts the cross-sectional mean of ``me`` within each date ×
    Fama-French 49-industry group.  The industry mean is computed over
    the **full CRSP universe** (not just the queried tickers) via a
    separate SQL query so that the adjustment is meaningful even when
    only a handful of stocks are requested.

    Reference: Asness, Porter & Stevens (2000).
    """
    from .industry_chars import _map_sic_to_industry  # local to avoid circular

    panel = raw_tables["__panel__"]
    engine = raw_tables.get("__engine__")

    if engine is None or engine.wrds_conn is None:
        logger.warning("me_ia: no engine reference — falling back to panel-only mean")
        industry_mean = panel.groupby(["date", "ind49"])["me"].transform("mean")
        return panel["me"] - industry_mean

    # ── Query full-universe prc, shrout, cfacpr, cfacshr, hsiccd ────────
    dates = pd.to_datetime(panel["date"])
    start_str = dates.min().strftime("%Y-%m-%d")
    end_str = dates.max().strftime("%Y-%m-%d")

    sf_table = "crsp.msf" if freq == "M" else "crsp.dsf"
    se_table = "crsp.mseall" if freq == "M" else "crsp.dseall"

    sql = (
        f"SELECT a.date, a.permco, "
        f"ABS(a.prc) / NULLIF(a.cfacpr, 0) AS prc, "
        f"a.shrout * a.cfacshr AS shrout, "
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
    logger.debug("me_ia universe SQL: %s", sql)
    try:
        univ = engine.wrds_conn.raw_sql(sql)
    except Exception:
        logger.error("me_ia: universe query failed — falling back", exc_info=True)
        industry_mean = panel.groupby(["date", "ind49"])["me"].transform("mean")
        return panel["me"] - industry_mean

    if univ.empty:
        return pd.Series(np.nan, index=panel.index)

    univ["date"] = pd.to_datetime(univ["date"])
    # Snap to month-end so MSF and panel dates align even when the
    # last trading day differs from the calendar month-end.
    if freq == "M":
        univ["date"] = univ["date"] + pd.offsets.MonthEnd(0)
    univ["me_univ"] = univ["prc"] * univ["shrout"]
    univ["ind49"] = _map_sic_to_industry(univ["hsiccd"], 49)

    # Cross-sectional mean ME by (date, ind49)
    ind_mean = (
        univ.groupby(["date", "ind49"])["me_univ"]
        .mean()
        .reset_index()
        .rename(columns={"me_univ": "__ind_mean__"})
    )

    # Merge the industry mean back onto the panel
    panel_tmp = panel[["date", "ind49", "me"]].copy()
    panel_tmp["date"] = pd.to_datetime(panel_tmp["date"])
    if freq == "M":
        panel_tmp["date"] = panel_tmp["date"] + pd.offsets.MonthEnd(0)
    merged = panel_tmp.merge(ind_mean, on=["date", "ind49"], how="left")

    return pd.Series(
        (merged["me"].values - merged["__ind_mean__"].values),
        index=panel.index,
    )

me_ia.needs = {"crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"],
               "crsp.seall": ["hsiccd"]}
me_ia._output_name = "me_ia"
me_ia._order = 9
me_ia._requires = ["me", "ind49"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers for new characteristics
# ═══════════════════════════════════════════════════════════════════════════

def _daily_only(panel: pd.DataFrame, freq: str) -> pd.Series | None:
    """Return a NaN series if *freq* is not daily, else ``None``."""
    if freq != "D":
        return pd.Series(np.nan, index=panel.index)
    return None


def _monthly_only_guard(panel: pd.DataFrame, freq: str) -> pd.Series | None:
    """Return a NaN series if *freq* is not monthly, else ``None``."""
    if freq != "M":
        return pd.Series(np.nan, index=panel.index)
    return None


_FF_CACHE: dict = {}


def _get_ff_data(dataset_name: str, start, end) -> pd.DataFrame | None:
    """Fetch Fama-French factor data via pandas_datareader, with caching."""
    key = (dataset_name, str(start)[:10], str(end)[:10])
    if key in _FF_CACHE:
        return _FF_CACHE[key]
    try:
        import pandas_datareader.data as pdr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ff = pdr.DataReader(dataset_name, "famafrench", start=start, end=end)
        df = ff[0] / 100.0  # percentage to decimal
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        _FF_CACHE[key] = df
        return df
    except Exception:
        logger.warning("Could not fetch FF data: %s", dataset_name, exc_info=True)
        return None


def _rolling_ols_residvar(y: np.ndarray, X: np.ndarray,
                          window: int) -> np.ndarray:
    """Compute rolling OLS residual variance over *y ~ X* (no intercept added).

    An intercept column must already be included in *X* if desired.
    Returns an array of length ``len(y)`` with NaN for positions where
    the window is incomplete or the regression cannot be estimated.
    """
    n = len(y)
    result = np.full(n, np.nan)
    k = X.shape[1]
    for i in range(window - 1, n):
        yi = y[i - window + 1:i + 1]
        Xi = X[i - window + 1:i + 1]
        valid = ~(np.isnan(yi) | np.any(np.isnan(Xi), axis=1))
        nv = int(valid.sum())
        if nv < k + 2:
            continue
        yi_v = yi[valid]
        Xi_v = Xi[valid]
        try:
            beta = np.linalg.lstsq(Xi_v, yi_v, rcond=None)[0]
            eps = yi_v - Xi_v @ beta
            result[i] = np.sum(eps ** 2) / (nv - k)
        except np.linalg.LinAlgError:
            pass
    return result


# ═══════════════════════════════════════════════════════════════════════════
# bas  (order 11)  —  Chung & Zhang (2014)
# ═══════════════════════════════════════════════════════════════════════════

def bas(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Bid-ask spread.

    .. math::
        \texttt{spread}_s = \frac{\texttt{askhi}_s - \texttt{bidlo}_s}
                                 {(\texttt{askhi}_s + \texttt{bidlo}_s)/2}

    For monthly frequency the spread is computed from each day's
    ask/high and bid/low in CRSP.DSF and then averaged within the
    month.  For daily frequency the spread is computed directly from
    the panel.

    Reference: Chung & Zhang (2014).
    """
    panel = raw_tables["__panel__"]

    if freq == "D":
        askhi = panel["askhi"].astype(float)
        bidlo = panel["bidlo"].astype(float)
        mid = (askhi + bidlo) / 2
        spread = (askhi - bidlo) / mid
        return spread.where(mid > 0, np.nan)

    # ── Monthly: query DSF for daily data and average within month ─────
    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        logger.warning("bas: no engine — falling back to panel askhi/bidlo")
        askhi = panel["askhi"].astype(float)
        bidlo = panel["bidlo"].astype(float)
        mid = (askhi + bidlo) / 2
        return ((askhi - bidlo) / mid).where(mid > 0, np.nan)

    dates = pd.to_datetime(panel["date"])
    start_str = dates.min().strftime("%Y-%m-%d")
    end_str = dates.max().strftime("%Y-%m-%d")
    permcos = panel["permco"].dropna().astype(int).unique().tolist()
    if not permcos:
        return pd.Series(np.nan, index=panel.index)

    permco_str = ", ".join(str(p) for p in permcos)
    sql = (
        f"SELECT date, permco, askhi, bidlo FROM crsp.dsf "
        f"WHERE date BETWEEN '{start_str}' AND '{end_str}' "
        f"AND permco IN ({permco_str})"
    )
    try:
        dsf = engine.wrds_conn.raw_sql(sql)
    except Exception:
        logger.error("bas: DSF query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    if dsf.empty:
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["askhi"] = pd.to_numeric(dsf["askhi"], errors="coerce")
    dsf["bidlo"] = pd.to_numeric(dsf["bidlo"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")
    mid = (dsf["askhi"] + dsf["bidlo"]) / 2
    dsf["spread"] = np.where(mid > 0, (dsf["askhi"] - dsf["bidlo"]) / mid, np.nan)
    dsf["month_end"] = dsf["date"] + pd.offsets.MonthEnd(0)

    avg_spread = (
        dsf.groupby(["permco", "month_end"])["spread"]
        .mean()
        .reset_index()
        .rename(columns={"month_end": "date", "spread": "__bas__"})
    )

    panel_dates = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
    merge_key = pd.DataFrame({"permco": panel["permco"].values,
                               "date": panel_dates.values})
    merge_key["permco"] = merge_key["permco"].astype("Int64")
    avg_spread["permco"] = avg_spread["permco"].astype("Int64")
    avg_spread["date"] = pd.to_datetime(avg_spread["date"])
    merged = merge_key.merge(avg_spread, on=["permco", "date"], how="left")
    return pd.Series(merged["__bas__"].values, index=panel.index)


bas.needs = {"crsp.sf": ["askhi", "bidlo"]}
bas._output_name = "bas"
bas._order = 11


# ═══════════════════════════════════════════════════════════════════════════
# bas_r3m  (order 12)  —  Amihud (1989)
# ═══════════════════════════════════════════════════════════════════════════

def bas_r3m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 3-month (or 63-day) average of bid-ask spread.

    .. math::
        \texttt{bas\_r3m}_t = \frac{1}{3}\sum_{\tau=0}^{2}\texttt{bas}_{t-\tau}

    For daily the computation uses a 63-day rolling window.

    Reference: Amihud (1989).
    """
    panel = raw_tables["__panel__"]
    window = 3 if freq == "M" else 63
    min_p = 2 if freq == "M" else 21
    return panel.groupby("permco")["bas"].transform(
        lambda x: x.rolling(window, min_periods=min_p).mean()
    )


bas_r3m.needs = {"crsp.sf": ["askhi", "bidlo"]}
bas_r3m._output_name = "bas_r3m"
bas_r3m._order = 12
bas_r3m._requires = ["bas"]


# ═══════════════════════════════════════════════════════════════════════════
# beta_r3m  (order 13)  —  Fama & MacBeth (1973)
# ═══════════════════════════════════════════════════════════════════════════

def beta_r3m(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 3-month CAPM beta.

    .. math::
        \beta_t = \frac{\operatorname{Cov}(r, r^m)}
                       {\operatorname{Var}(r^m)},
        \quad \text{over a rolling window ending at } t

    For daily the computation uses a 63-day rolling window.

    Reference: Fama & MacBeth (1973).
    """
    panel = raw_tables["__panel__"]
    window = 3 if freq == "M" else 63
    min_p = 3 if freq == "M" else 21

    def _beta_per_group(group):
        ret = group["ret"].astype(float)
        mkt = group["vwretd"].astype(float)
        cov = ret.rolling(window, min_periods=min_p).cov(mkt)
        var = mkt.rolling(window, min_periods=min_p).var()
        return cov / var

    return panel.groupby("permco", group_keys=False).apply(_beta_per_group)


beta_r3m.needs = {"crsp.sf": ["ret"], "crsp.si": ["vwretd"]}
beta_r3m._output_name = "beta_r3m"
beta_r3m._order = 13


# ═══════════════════════════════════════════════════════════════════════════
# dvol  (order 14)  —  Chordia et al. (2001)
# ═══════════════════════════════════════════════════════════════════════════

def dvol(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Log dollar volume (2-period lag).

    .. math::
        \texttt{dvol}_t = \ln(\texttt{vol}_{t-2} \times \texttt{prc}_{t-2})

    Reference: Chordia, Subrahmanyam & Anshuman (2001).
    """
    panel = raw_tables["__panel__"]
    lag_n = _lag(freq, 2)

    # Recover absolute raw price: adj_prc = |raw_prc| / cfacpr
    raw_prc = (panel["prc"].astype(float)
               * panel["cfacpr"].astype(float))
    vol = panel["vol"].astype(float)

    raw_prc_lag = panel.assign(__rp=raw_prc).groupby("permco")["__rp"].shift(lag_n)
    vol_lag = panel.groupby("permco")["vol"].shift(lag_n).astype(float)

    dollar_vol = vol_lag * raw_prc_lag
    return pd.Series(
        np.where(dollar_vol > 0, np.log(dollar_vol), np.nan),
        index=panel.index,
    )


dvol.needs = {"crsp.sf": ["prc", "vol", "cfacpr"]}
dvol._output_name = "dvol"
dvol._order = 14
dvol._requires = ["prc"]


# ═══════════════════════════════════════════════════════════════════════════
# dy  (order 15)  —  Litzenberger & Ramaswamy (1982)
# ═══════════════════════════════════════════════════════════════════════════

def dy(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Dividend yield.

    .. math::
        \texttt{dy}_t = \frac{1}{\texttt{me}_t}
                        \sum_{\tau=-11}^{0}
                        (\texttt{ret}_{t+\tau} - \texttt{retx}_{t+\tau})
                        \,\texttt{me}_{t+\tau-1}

    For daily the computation uses a 252-day rolling window.

    Reference: Litzenberger & Ramaswamy (1982).
    """
    panel = raw_tables["__panel__"]
    window = 12 if freq == "M" else 252

    div_ret = panel["ret"].astype(float) - panel["retx"].astype(float)
    me_lag = panel.groupby("permco")["me"].shift(1).astype(float)
    div_dollar = div_ret * me_lag

    rolling_div = panel.assign(__dd=div_dollar).groupby("permco")["__dd"].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    me_cur = panel["me"].astype(float)
    return pd.Series(
        np.where(me_cur != 0, rolling_div.values / me_cur.values, np.nan),
        index=panel.index,
    )


dy.needs = {"crsp.sf": ["ret", "retx", "prc", "cfacpr", "shrout", "cfacshr"]}
dy._output_name = "dy"
dy._order = 15
dy._requires = ["me"]


# ═══════════════════════════════════════════════════════════════════════════
# illiq  (order 16)  —  Amihud (2002)
# ═══════════════════════════════════════════════════════════════════════════

def illiq(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Amihud illiquidity (daily only).

    .. math::
        \texttt{illiq}_t = \frac{1}{252}
            \sum_{\tau=-251}^{0}
            \frac{|\texttt{ret}_{t+\tau}|}
                 {\texttt{prc}_{t+\tau} \times \texttt{vol}_{t+\tau}}

    Reference: Amihud (2002).
    """
    panel = raw_tables["__panel__"]
    guard = _daily_only(panel, freq)
    if guard is not None:
        return guard

    ret_abs = panel["ret"].astype(float).abs()
    # Use raw absolute price for dollar volume
    raw_prc = (panel["prc"].astype(float)
               * panel["cfacpr"].astype(float))
    vol = panel["vol"].astype(float)
    dollar_vol = raw_prc * vol

    ratio = pd.Series(
        np.where(dollar_vol > 0, ret_abs.values / dollar_vol.values, np.nan),
        index=panel.index,
    )

    return panel.assign(__ratio=ratio).groupby("permco")["__ratio"].transform(
        lambda x: x.rolling(252, min_periods=60).mean()
    )


illiq.needs = {"crsp.sf": ["ret", "prc", "vol", "cfacpr"]}
illiq._output_name = "illiq"
illiq._order = 16
illiq._requires = ["prc"]


# ═══════════════════════════════════════════════════════════════════════════
# turn  (order 23)  —  Datar et al. (1998)
# ═══════════════════════════════════════════════════════════════════════════

def turn(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Share turnover.

    .. math:: \texttt{turn}_t = \frac{\texttt{vol}_t}{\texttt{shrout}_t}

    Uses raw (unadjusted) volume and shares outstanding so that the
    ratio is comparable across stock-split events.

    Reference: Datar, Naik & Radcliffe (1998).
    """
    panel = raw_tables["__panel__"]
    vol = panel["vol"].astype(float)
    # Recover raw shrout: adj_shrout = raw_shrout × cfacshr
    adj_shrout = panel["shrout"].astype(float)
    cfacshr = panel["cfacshr"].astype(float)
    raw_shrout = np.where(cfacshr != 0, adj_shrout / cfacshr, np.nan)
    return pd.Series(
        np.where(raw_shrout != 0, vol.values / raw_shrout, np.nan),
        index=panel.index,
    )


turn.needs = {"crsp.sf": ["vol", "shrout", "cfacshr"]}
turn._output_name = "turn"
turn._order = 23
turn._requires = ["shrout"]


# ═══════════════════════════════════════════════════════════════════════════
# rvar_mean  (order 21)
# ═══════════════════════════════════════════════════════════════════════════

def rvar_mean(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 63-day return variance (daily only).

    .. math::
        \texttt{rvar\_mean}_t = \operatorname{Var}(\texttt{ret}_s),
        \quad s \in [t-63d,\,t]
    """
    panel = raw_tables["__panel__"]
    guard = _daily_only(panel, freq)
    if guard is not None:
        return guard

    return panel.groupby("permco")["ret"].transform(
        lambda x: x.astype(float).rolling(63, min_periods=21).var()
    )


rvar_mean.needs = {"crsp.sf": ["ret"]}
rvar_mean._output_name = "rvar_mean"
rvar_mean._order = 21


# ═══════════════════════════════════════════════════════════════════════════
# std_dvol  (order 22)  —  Chordia et al. (2001)
# ═══════════════════════════════════════════════════════════════════════════

def std_dvol(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 63-day std of log dollar volume (daily only).

    .. math::
        \texttt{std\_dvol}_t = \sigma(\texttt{dvol}_s),
        \quad s \in [t-63d,\,t]

    Reference: Chordia, Subrahmanyam & Anshuman (2001).
    """
    panel = raw_tables["__panel__"]
    guard = _daily_only(panel, freq)
    if guard is not None:
        return guard

    return panel.groupby("permco")["dvol"].transform(
        lambda x: x.astype(float).rolling(63, min_periods=21).std()
    )


std_dvol.needs = {"crsp.sf": ["prc", "vol", "cfacpr"]}
std_dvol._output_name = "std_dvol"
std_dvol._order = 22
std_dvol._requires = ["dvol"]


# ═══════════════════════════════════════════════════════════════════════════
# std_turn  (order 24)  —  Chordia et al. (2001)
# ═══════════════════════════════════════════════════════════════════════════

def std_turn(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 63-day std of share turnover (daily only).

    .. math::
        \texttt{std\_turn}_t = \sigma(\texttt{turn}_s),
        \quad s \in [t-63d,\,t]

    Reference: Chordia, Subrahmanyam & Anshuman (2001).
    """
    panel = raw_tables["__panel__"]
    guard = _daily_only(panel, freq)
    if guard is not None:
        return guard

    return panel.groupby("permco")["turn"].transform(
        lambda x: x.astype(float).rolling(63, min_periods=21).std()
    )


std_turn.needs = {"crsp.sf": ["vol", "shrout", "cfacshr"]}
std_turn._output_name = "std_turn"
std_turn._order = 24
std_turn._requires = ["turn"]


# ═══════════════════════════════════════════════════════════════════════════
# zerotrade  (order 25)  —  Liu (2006)
# ═══════════════════════════════════════════════════════════════════════════

def zerotrade(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 63-day zero-trading-days measure (daily only).

    .. math::
        \texttt{zerotrade}_t = N_{\text{zero}}
            + \frac{1}{3\,\overline{\texttt{turn}}}\,\frac{1}{480000}

    where :math:`N_{\text{zero}}` is the number of zero-volume days
    and :math:`\overline{\texttt{turn}}` is the average daily turnover
    over the past 63 trading days.

    Reference: Liu (2006).
    """
    panel = raw_tables["__panel__"]
    guard = _daily_only(panel, freq)
    if guard is not None:
        return guard

    vol = panel["vol"].astype(float)
    is_zero = (vol == 0).astype(float)
    turn_vals = panel["turn"].astype(float)

    def _per_group(g):
        z = g["__is_zero"]
        t = g["__turn"]
        n_zero = z.rolling(63, min_periods=21).sum()
        avg_turn = t.rolling(63, min_periods=21).mean()
        return n_zero + (1.0 / (3.0 * avg_turn)) * (1.0 / 480000.0)

    return (
        panel.assign(__is_zero=is_zero, __turn=turn_vals)
        .groupby("permco", group_keys=False)
        .apply(_per_group)
    )


zerotrade.needs = {"crsp.sf": ["vol", "shrout", "cfacshr"]}
zerotrade._output_name = "zerotrade"
zerotrade._order = 25
zerotrade._requires = ["turn"]


# ═══════════════════════════════════════════════════════════════════════════
# rvar_capm  (order 17)
# ═══════════════════════════════════════════════════════════════════════════

def rvar_capm(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 3-month CAPM residual variance.

    .. math::
        \texttt{rvar\_capm}_t = \operatorname{Var}(\hat\varepsilon_s),
        \quad s \in [t-3\text{m},\,t]

    For daily the computation uses a 63-day rolling window.
    For monthly the engine queries CRSP.DSF daily returns.
    """
    panel = raw_tables["__panel__"]

    if freq == "D":
        window = 63
        ret = panel["ret"].astype(float).values
        mkt = panel["vwretd"].astype(float).values

        def _per_group(g):
            y = g["ret"].astype(float).values
            x = g["vwretd"].astype(float).values
            X = np.column_stack([np.ones(len(y)), x])
            return pd.Series(
                _rolling_ols_residvar(y, X, window), index=g.index
            )

        return panel.groupby("permco", group_keys=False).apply(_per_group)

    # ── Monthly: query DSF for daily data ──────────────────────────────
    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        return pd.Series(np.nan, index=panel.index)

    dates = pd.to_datetime(panel["date"])
    start = (dates.min() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")
    permcos = panel["permco"].dropna().astype(int).unique().tolist()
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
        logger.error("rvar_capm: DSF/DSI query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")
    dsi["date"] = pd.to_datetime(dsi["date"])
    dsf = dsf.merge(dsi, on="date", how="left")
    dsf = dsf.sort_values(["permco", "date"])
    dsf["month_end"] = dsf["date"] + pd.offsets.MonthEnd(0)

    results: dict = {}
    for pc, grp in dsf.groupby("permco"):
        y = grp["ret"].values.astype(float)
        x = grp["vwretd"].values.astype(float)
        X = np.column_stack([np.ones(len(y)), x])
        rv = _rolling_ols_residvar(y, X, 63)
        me = grp["month_end"].values
        for m_end in np.unique(me):
            mask = me == m_end
            vals = rv[mask]
            last = vals[~np.isnan(vals)]
            if len(last):
                results[(pc, m_end)] = last[-1]

    panel_me = (pd.to_datetime(panel["date"])
                + pd.offsets.MonthEnd(0)).values
    out = np.full(len(panel), np.nan)
    for i in range(len(panel)):
        key = (panel.iloc[i]["permco"], panel_me[i])
        if key in results:
            out[i] = results[key]
    return pd.Series(out, index=panel.index)


rvar_capm.needs = {"crsp.sf": ["ret"], "crsp.si": ["vwretd"]}
rvar_capm._output_name = "rvar_capm"
rvar_capm._order = 17


# ═══════════════════════════════════════════════════════════════════════════
# rvar_ff3  (order 18)  —  Fama & French (1993)
# ═══════════════════════════════════════════════════════════════════════════

def _rvar_factor_model(raw_tables, freq, ff_dataset_daily, ff_dataset_monthly,
                       factor_cols):
    """Shared implementation for factor-model residual variance chars."""
    panel = raw_tables["__panel__"]

    if freq == "D":
        dates = pd.to_datetime(panel["date"])
        dataset = ff_dataset_daily
        ff = _get_ff_data(dataset, dates.min() - pd.DateOffset(months=1),
                          dates.max())
        if ff is None:
            return pd.Series(np.nan, index=panel.index)
        # Rename columns to avoid special characters
        col_map = {c: c.strip().replace("-", "_") for c in ff.columns}
        ff = ff.rename(columns=col_map)
        mapped_factor_cols = [col_map.get(c, c) for c in factor_cols]
        # Merge FF data onto panel
        panel_tmp = panel[["date", "permco", "ret"]].copy()
        panel_tmp["date"] = pd.to_datetime(panel_tmp["date"])
        panel_tmp = panel_tmp.merge(ff, left_on="date", right_index=True,
                                    how="left")

        def _per_group(g):
            y = g["ret"].astype(float).values
            X_data = g[mapped_factor_cols].astype(float).values
            X = np.column_stack([np.ones(len(y)), X_data])
            return pd.Series(
                _rolling_ols_residvar(y, X, 63), index=g.index
            )

        return panel_tmp.groupby("permco", group_keys=False).apply(_per_group)

    # ── Monthly: query DSF and merge FF daily factors ──────────────────
    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        return pd.Series(np.nan, index=panel.index)

    dates = pd.to_datetime(panel["date"])
    start = (dates.min() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")
    permcos = panel["permco"].dropna().astype(int).unique().tolist()
    if not permcos:
        return pd.Series(np.nan, index=panel.index)

    permco_str = ", ".join(str(p) for p in permcos)
    try:
        dsf = engine.wrds_conn.raw_sql(
            f"SELECT date, permco, ret FROM crsp.dsf "
            f"WHERE date BETWEEN '{start}' AND '{end}' "
            f"AND permco IN ({permco_str})"
        )
    except Exception:
        logger.error("rvar factor: DSF query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")

    ff = _get_ff_data(ff_dataset_daily,
                      pd.to_datetime(start) - pd.DateOffset(months=1),
                      pd.to_datetime(end))
    if ff is None:
        return pd.Series(np.nan, index=panel.index)

    col_map = {c: c.strip().replace("-", "_") for c in ff.columns}
    ff = ff.rename(columns=col_map)
    mapped_factor_cols = [col_map.get(c, c) for c in factor_cols]

    dsf = dsf.merge(ff, left_on="date", right_index=True, how="left")
    dsf = dsf.sort_values(["permco", "date"])
    dsf["month_end"] = dsf["date"] + pd.offsets.MonthEnd(0)

    results: dict = {}
    for pc, grp in dsf.groupby("permco"):
        y = grp["ret"].values.astype(float)
        X_data = grp[mapped_factor_cols].values.astype(float)
        X = np.column_stack([np.ones(len(y)), X_data])
        rv = _rolling_ols_residvar(y, X, 63)
        me = grp["month_end"].values
        for m_end in np.unique(me):
            mask = me == m_end
            vals = rv[mask]
            last = vals[~np.isnan(vals)]
            if len(last):
                results[(pc, m_end)] = last[-1]

    panel_me = (pd.to_datetime(panel["date"])
                + pd.offsets.MonthEnd(0)).values
    out = np.full(len(panel), np.nan)
    for i in range(len(panel)):
        key = (panel.iloc[i]["permco"], panel_me[i])
        if key in results:
            out[i] = results[key]
    return pd.Series(out, index=panel.index)


def rvar_ff3(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 3-month Fama-French three-factor residual variance.

    .. math::
        \texttt{rvar\_ff3}_t = \operatorname{Var}(\hat\varepsilon_s),
        \quad s \in [t-3\text{m},\,t]

    Reference: Fama & French (1993).
    """
    return _rvar_factor_model(
        raw_tables, freq,
        ff_dataset_daily="F-F_Research_Data_Factors_daily",
        ff_dataset_monthly="F-F_Research_Data_Factors",
        factor_cols=["Mkt-RF", "SMB", "HML"],
    )


rvar_ff3.needs = {"crsp.sf": ["ret"]}
rvar_ff3._output_name = "rvar_ff3"
rvar_ff3._order = 18


# ═══════════════════════════════════════════════════════════════════════════
# rvar_ff5  (order 19)  —  Fama & French (2015)
# ═══════════════════════════════════════════════════════════════════════════

def rvar_ff5(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 3-month Fama-French five-factor residual variance.

    Reference: Fama & French (2015).
    """
    return _rvar_factor_model(
        raw_tables, freq,
        ff_dataset_daily="F-F_Research_Data_5_Factors_2x3_daily",
        ff_dataset_monthly="F-F_Research_Data_5_Factors_2x3",
        factor_cols=["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    )


rvar_ff5.needs = {"crsp.sf": ["ret"]}
rvar_ff5._output_name = "rvar_ff5"
rvar_ff5._order = 19


# ═══════════════════════════════════════════════════════════════════════════
# rvar_car  (order 20)  —  Carhart (1997)
# ═══════════════════════════════════════════════════════════════════════════

def rvar_car(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Rolling 3-month Carhart four-factor residual variance.

    Reference: Carhart (1997).
    """
    panel = raw_tables["__panel__"]

    # Carhart = FF3 + Momentum.  We need to fetch and merge both datasets.
    dates = pd.to_datetime(panel["date"])
    start = dates.min() - pd.DateOffset(months=5)
    end = dates.max()

    if freq == "D":
        ff3 = _get_ff_data("F-F_Research_Data_Factors_daily", start, end)
        mom = _get_ff_data("F-F_Momentum_Factor_daily", start, end)
    else:
        ff3 = _get_ff_data("F-F_Research_Data_Factors_daily", start, end)
        mom = _get_ff_data("F-F_Momentum_Factor_daily", start, end)

    if ff3 is None or mom is None:
        return pd.Series(np.nan, index=panel.index)

    # Merge the two datasets
    combined = ff3.join(mom, how="inner")
    col_map = {c: c.strip().replace("-", "_") for c in combined.columns}
    combined = combined.rename(columns=col_map)

    # Identify the momentum column (may be 'Mom' or 'Mom   ')
    mom_col = [c for c in combined.columns if "mom" in c.lower()]
    if not mom_col:
        # Fallback: try the original column name
        mom_col = [c for c in combined.columns if c not in col_map.values()
                   or "Mom" in c]
    factor_cols_mapped = ["Mkt_RF", "SMB", "HML"] + mom_col[:1]

    if freq == "D":
        panel_tmp = panel[["date", "permco", "ret"]].copy()
        panel_tmp["date"] = pd.to_datetime(panel_tmp["date"])
        panel_tmp = panel_tmp.merge(combined, left_on="date",
                                    right_index=True, how="left")

        def _per_group(g):
            y = g["ret"].astype(float).values
            X_data = g[factor_cols_mapped].astype(float).values
            X = np.column_stack([np.ones(len(y)), X_data])
            return pd.Series(
                _rolling_ols_residvar(y, X, 63), index=g.index
            )

        return panel_tmp.groupby("permco", group_keys=False).apply(_per_group)

    # ── Monthly: query DSF ─────────────────────────────────────────────
    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        return pd.Series(np.nan, index=panel.index)

    start_str = (dates.min() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    end_str = dates.max().strftime("%Y-%m-%d")
    permcos = panel["permco"].dropna().astype(int).unique().tolist()
    if not permcos:
        return pd.Series(np.nan, index=panel.index)

    permco_str = ", ".join(str(p) for p in permcos)
    try:
        dsf = engine.wrds_conn.raw_sql(
            f"SELECT date, permco, ret FROM crsp.dsf "
            f"WHERE date BETWEEN '{start_str}' AND '{end_str}' "
            f"AND permco IN ({permco_str})"
        )
    except Exception:
        logger.error("rvar_car: DSF query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")
    dsf = dsf.merge(combined, left_on="date", right_index=True, how="left")
    dsf = dsf.sort_values(["permco", "date"])
    dsf["month_end"] = dsf["date"] + pd.offsets.MonthEnd(0)

    results: dict = {}
    for pc, grp in dsf.groupby("permco"):
        y = grp["ret"].values.astype(float)
        X_data = grp[factor_cols_mapped].values.astype(float)
        X = np.column_stack([np.ones(len(y)), X_data])
        rv = _rolling_ols_residvar(y, X, 63)
        me = grp["month_end"].values
        for m_end in np.unique(me):
            mask = me == m_end
            vals = rv[mask]
            last = vals[~np.isnan(vals)]
            if len(last):
                results[(pc, m_end)] = last[-1]

    panel_me = (pd.to_datetime(panel["date"])
                + pd.offsets.MonthEnd(0)).values
    out = np.full(len(panel), np.nan)
    for i in range(len(panel)):
        key = (panel.iloc[i]["permco"], panel_me[i])
        if key in results:
            out[i] = results[key]
    return pd.Series(out, index=panel.index)


rvar_car.needs = {"crsp.sf": ["ret"]}
rvar_car._output_name = "rvar_car"
rvar_car._order = 20


# ═══════════════════════════════════════════════════════════════════════════
# psliq  (order 26)  —  Pastor & Stambaugh (2003)
# ═══════════════════════════════════════════════════════════════════════════

def psliq(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    r"""Pastor-Stambaugh liquidity index (monthly only).

    For each stock *i* in month *m*, estimate via OLS on daily data:

    .. math::
        r_{i,d} = \theta_{i,m} + \phi_{1,i,m}\,r_{i,d-1}
                  + \phi_{2,i,m}\,r_{i,d-1}\,\texttt{dvol}_{i,d-1}
                  + \varepsilon_{i,d}

    The liquidity measure is :math:`\phi_{2,i,m}`, estimated using daily
    data within each calendar month.

    Reference: Pastor & Stambaugh (2003).
    """
    panel = raw_tables["__panel__"]
    guard = _monthly_only_guard(panel, freq)
    if guard is not None:
        return guard

    engine = raw_tables.get("__engine__")
    if engine is None or engine.wrds_conn is None:
        logger.warning("psliq: no engine — returning NaN")
        return pd.Series(np.nan, index=panel.index)

    dates = pd.to_datetime(panel["date"])
    start_str = (dates.min() - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    end_str = dates.max().strftime("%Y-%m-%d")
    permcos = panel["permco"].dropna().astype(int).unique().tolist()
    if not permcos:
        return pd.Series(np.nan, index=panel.index)

    permco_str = ", ".join(str(p) for p in permcos)
    try:
        dsf = engine.wrds_conn.raw_sql(
            f"SELECT date, permco, ret, prc, vol FROM crsp.dsf "
            f"WHERE date BETWEEN '{start_str}' AND '{end_str}' "
            f"AND permco IN ({permco_str})"
        )
    except Exception:
        logger.error("psliq: DSF query failed", exc_info=True)
        return pd.Series(np.nan, index=panel.index)

    if dsf.empty:
        return pd.Series(np.nan, index=panel.index)

    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    dsf["prc"] = pd.to_numeric(dsf["prc"], errors="coerce").abs()
    dsf["vol"] = pd.to_numeric(dsf["vol"], errors="coerce")
    dsf["permco"] = dsf["permco"].astype("Int64")
    dsf = dsf.sort_values(["permco", "date"])
    dsf["month_end"] = dsf["date"] + pd.offsets.MonthEnd(0)

    # Lagged return and lagged dollar volume per permco
    dsf["ret_lag"] = dsf.groupby("permco")["ret"].shift(1)
    dsf["dvol_raw"] = dsf["prc"] * dsf["vol"]
    dsf["dvol_lag"] = dsf.groupby("permco")["dvol_raw"].shift(1)
    # Interaction: ret_{d-1} * dvol_{d-1}
    dsf["ret_dvol_lag"] = dsf["ret_lag"] * dsf["dvol_lag"]

    result_records: list[dict] = []
    for (pc, me), grp in dsf.groupby(["permco", "month_end"]):
        g = grp.dropna(subset=["ret", "ret_lag", "ret_dvol_lag"])
        if len(g) < 5:
            continue
        y = g["ret"].values.astype(float)
        X = np.column_stack([
            np.ones(len(y)),
            g["ret_lag"].values.astype(float),
            g["ret_dvol_lag"].values.astype(float),
        ])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            result_records.append({"permco": pc, "date": me, "__psliq__": beta[2]})
        except np.linalg.LinAlgError:
            pass

    if not result_records:
        return pd.Series(np.nan, index=panel.index)

    # Merge results onto panel via (permco, month-end date)
    result_df = pd.DataFrame(result_records)
    result_df["permco"] = result_df["permco"].astype("Int64")
    result_df["date"] = pd.to_datetime(result_df["date"]) + pd.offsets.MonthEnd(0)

    panel_tmp = panel[["permco", "date"]].copy()
    panel_tmp["date"] = pd.to_datetime(panel_tmp["date"]) + pd.offsets.MonthEnd(0)
    panel_tmp["permco"] = panel_tmp["permco"].astype("Int64")
    merged = panel_tmp.merge(result_df, on=["permco", "date"], how="left")
    return pd.Series(merged["__psliq__"].values, index=panel.index)


psliq.needs = {"crsp.sf": ["ret", "prc", "vol"]}
psliq._output_name = "psliq"
psliq._order = 26
