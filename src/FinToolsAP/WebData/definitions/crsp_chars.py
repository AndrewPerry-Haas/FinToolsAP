"""
FinToolsAP.WebData.definitions.crsp_chars
==========================================

Built-in characteristics derived from CRSP stock files (MSF / DSF)
and CRSP security-event files (MSEALL / DSEALL).

Convention
----------
* Each function receives ``(raw_tables: dict[str, DataFrame], freq: str)``
  and returns a ``pandas.Series`` aligned to the index of the panel.
* ``raw_tables`` keys are WRDS table aliases (e.g. ``'crsp.msf'``).
* The ``.needs`` attribute declares source dependencies.
* ``_output_name`` may alias the function to a different column name.
* ``_order`` controls execution priority (lower = earlier).

Identity columns (``ticker``, ``date``, ``permco``) are injected by the
engine and do **not** need characteristic definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Raw pass-through / light-cleaning characteristics
# ═══════════════════════════════════════════════════════════════════════════

# --- price (split-adjusted absolute value) --------------------------------

def prc(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Split-adjusted absolute stock price."""
    panel = raw_tables["__panel__"]
    adjusted = panel["prc"].abs() / panel["cfacpr"]
    return adjusted

prc.needs = {"crsp.sf": ["prc", "cfacpr"]}
prc._output_name = "prc"
prc._order = 10


# --- shares outstanding (split-adjusted, in millions) ---------------------

def shrout(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Split-adjusted shares outstanding (millions)."""
    panel = raw_tables["__panel__"]
    adjusted = (panel["shrout"] * panel["cfacshr"]) / 1e3
    return adjusted

shrout.needs = {"crsp.sf": ["shrout", "cfacshr"]}
shrout._output_name = "shrout"
shrout._order = 10


# --- return ---------------------------------------------------------------

def clean_ret(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Holding-period return, NaN filled to 0."""
    panel = raw_tables["__panel__"]
    return panel["ret"].fillna(0)

clean_ret.needs = {"crsp.sf": ["ret"]}
clean_ret._output_name = "ret"
clean_ret._order = 5


# --- return ex-dividends --------------------------------------------------

def clean_retx(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Holding-period return excluding dividends, NaN filled to 0."""
    panel = raw_tables["__panel__"]
    return panel["retx"].fillna(0)

clean_retx.needs = {"crsp.sf": ["retx"]}
clean_retx._output_name = "retx"
clean_retx._order = 5


# --- volume ---------------------------------------------------------------

def vol(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Trading volume."""
    panel = raw_tables["__panel__"]
    return panel["vol"]

vol.needs = {"crsp.sf": ["vol"]}
vol._output_name = "vol"
vol._order = 5


# --- bid-low (split-adjusted) --------------------------------------------

def bidlo(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Split-adjusted low price (monthly low or daily low)."""
    panel = raw_tables["__panel__"]
    return panel["bidlo"].abs() / panel["cfacpr"]

bidlo.needs = {"crsp.sf": ["bidlo", "cfacpr"]}
bidlo._output_name = "bidlo"
bidlo._order = 10


# --- ask-high (split-adjusted) -------------------------------------------

def askhi(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Split-adjusted high price (monthly high or daily high)."""
    panel = raw_tables["__panel__"]
    return panel["askhi"].abs() / panel["cfacpr"]

askhi.needs = {"crsp.sf": ["askhi", "cfacpr"]}
askhi._output_name = "askhi"
askhi._order = 10


# ═══════════════════════════════════════════════════════════════════════════
# Derived characteristics
# ═══════════════════════════════════════════════════════════════════════════

# --- market equity --------------------------------------------------------

def me(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Market equity in millions  =  prc × shrout.

    Both ``prc`` and ``shrout`` are already split-adjusted by their
    respective order-10 characteristics, so no further cfacpr/cfacshr
    adjustment is needed here.
    """
    panel = raw_tables["__panel__"]
    return panel["prc"] * panel["shrout"]

me.needs = {"crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"]}
me._output_name = "me"
me._order = 20
me._requires = ["prc", "shrout"]


# --- dividend amount ------------------------------------------------------

def div(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Per-share dividend  =  (ret − retx) × lagged price.

    ``ret``/``retx`` are already cleaned (order 5) and ``prc`` is already
    split-adjusted (order 10), so we just lag the adjusted price.
    """
    panel = raw_tables["__panel__"]
    ret = panel["ret"]
    retx = panel["retx"]
    lagged_prc = panel.groupby("permco")["prc"].shift(1)
    dividend = (ret - retx) * lagged_prc
    return dividend.fillna(0)

div.needs = {"crsp.sf": ["ret", "retx", "prc", "cfacpr", "permco"]}
div._output_name = "div"
div._order = 30
div._requires = ["ret", "retx", "prc"]


# --- dividend yield (dp) -------------------------------------------------

def dp(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Dividend yield  =  rolling-12-month dividend sum / price.

    Uses a 12-period (monthly) or 252-period (daily) rolling window.
    ``prc`` is already split-adjusted (order 10); ``ret``/``retx`` are
    already cleaned (order 5).
    """
    panel = raw_tables["__panel__"]
    window = 12 if freq == "M" else 252
    min_per = 7 if freq == "M" else 147

    ret = panel["ret"]
    retx = panel["retx"]
    prc_val = panel["prc"]
    lagged_prc = panel.groupby("permco")["prc"].shift(1)
    dividend = ((ret - retx) * lagged_prc).fillna(0)

    div_12m = panel.assign(__div=dividend).groupby("permco")["__div"].transform(
        lambda x: x.rolling(window=window, min_periods=min_per).sum()
    )
    prc_f = prc_val.astype(float)
    result = np.where(prc_f != 0, div_12m.astype(float) / prc_f, np.nan)
    return pd.Series(result, index=panel.index)

dp.needs = {"crsp.sf": ["ret", "retx", "prc", "cfacpr", "permco"]}
dp._output_name = "dp"
dp._order = 40
dp._requires = ["ret", "retx", "prc"]


# --- dividend per share (dps) --------------------------------------------

def dps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Dividend per share  =  rolling-12-month dividend sum / shrout.

    ``prc`` and ``shrout`` are already split-adjusted (order 10);
    ``ret``/``retx`` are already cleaned (order 5).
    """
    panel = raw_tables["__panel__"]
    window = 12 if freq == "M" else 252
    min_per = 7 if freq == "M" else 147

    ret = panel["ret"]
    retx = panel["retx"]
    lagged_prc = panel.groupby("permco")["prc"].shift(1)
    dividend = ((ret - retx) * lagged_prc).fillna(0)
    div_12m = panel.assign(__div=dividend).groupby("permco")["__div"].transform(
        lambda x: x.rolling(window=window, min_periods=min_per).sum()
    )
    shares = panel["shrout"].astype(float)
    result = np.where(shares != 0, div_12m.astype(float) / shares, np.nan)
    return pd.Series(result, index=panel.index)

dps.needs = {"crsp.sf": ["ret", "retx", "prc", "cfacpr", "shrout", "cfacshr", "permco"]}
dps._output_name = "dps"
dps._order = 40
dps._requires = ["ret", "retx", "prc", "shrout"]


# --- price per share (pps) -----------------------------------------------

def pps(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Price per share  =  prc / shrout (both already split-adjusted)."""
    panel = raw_tables["__panel__"]
    prc_f = panel["prc"].astype(float)
    shrout_f = panel["shrout"].astype(float)
    result = np.where(shrout_f != 0, prc_f / shrout_f, np.nan)
    return pd.Series(result, index=panel.index)

pps.needs = {"crsp.sf": ["prc", "cfacpr", "shrout", "cfacshr"]}
pps._output_name = "pps"
pps._order = 25
pps._requires = ["prc", "shrout"]
