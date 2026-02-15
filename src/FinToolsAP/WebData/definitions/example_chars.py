"""
FinToolsAP.WebData.definitions.example_chars
=============================================

Two reference examples showing how to define characteristics:

1. **``book_to_market``** – a standard cross-database characteristic that
   requires a join between CRSP (market equity) and Compustat (book equity).
2. **``clean_ret``** – a "cleaning" function that reads raw ``ret``, handles
   missing-value codes, and maps its output back to the name ``'ret'``
   (overwriting the raw column).

These are *also* registered as real characteristics – they are not stubs.
``clean_ret`` is identical to the one in ``crsp_chars.py`` and will
**overwrite** it (last-registered wins), demonstrating the aliasing /
override mechanism.

How to write your own
---------------------
1. Write a function with signature::

       def my_char(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
           ...

2. Attach a ``.needs`` dict mapping WRDS table aliases → column lists::

       my_char.needs = {'crsp.msf': ['prc', 'shrout']}

3. (Optional) set ``._output_name`` to alias the output column,
   and ``._order`` to control execution priority.

4. Drop the file in ``~/.fintoolsap/custom_chars.py`` (or in this
   ``definitions/`` directory for built-ins) and the registry will
   auto-discover it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Example 1:  Standard cross-database characteristic
# ═══════════════════════════════════════════════════════════════════════════

def book_to_market(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Book-to-market ratio  =  Book Equity / Market Equity.

    This characteristic demonstrates a cross-database join:
    * **Book equity** comes from Compustat FUNDQ
      (``seqq + txditcq − coalesce(pstkrv, pstkq, 0)``).
    * **Market equity** comes from CRSP stock files
      (``|prc| / cfacpr  ×  shrout × cfacshr / 1e3``).

    The engine fetches both tables, merges them via the CCM link table,
    and passes the unified ``raw_tables`` dict to this function.
    """
    # --- Book equity (Compustat, already merged onto panel) ---
    panel = raw_tables["__panel__"]
    seq   = panel["seqq"].fillna(0)
    txditc = panel["txditcq"].fillna(0)
    pstk  = panel["pstkrq"].fillna(panel["pstkq"]).fillna(0)
    book_equity = seq + txditc - pstk

    # --- Market equity (CRSP, already split-adjusted by order-10 chars) ---
    market_equity = panel["prc"] * panel["shrout"]

    # --- Ratio ---
    me_f = market_equity.astype(float)
    result = np.where(me_f != 0, book_equity.astype(float) / me_f, np.nan)
    return pd.Series(result, index=panel.index)


# Declare data requirements (the engine reads this before any SQL is sent)
book_to_market.needs = {
    "crsp.sf":     ["prc", "cfacpr", "shrout", "cfacshr"],
    "comp.fundq":  ["seqq", "txditcq", "pstkrq", "pstkq"],
}
# Optional metadata
book_to_market._output_name = "book_to_market"   # column name in final DF
book_to_market._order = 80                        # runs after prc/shrout/be
book_to_market._requires = ["prc", "shrout"]       # needs adjusted price & shares


# ═══════════════════════════════════════════════════════════════════════════
# Example 2:  Cleaning function (maps output to existing column name)
# ═══════════════════════════════════════════════════════════════════════════

def clean_return(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Clean raw holding-period returns.

    * Fills NaN / missing-value sentinel codes with 0.
    * Clips extreme outliers beyond ±200% in a single period.

    This function demonstrates the **aliasing** mechanism:
    its ``_output_name`` is set to ``'ret'``, so the cleaned series
    *replaces* the raw ``ret`` column in the final output rather than
    creating a separate ``'clean_return'`` column.

    .. note::
       Because the built-in ``crsp_chars.clean_ret`` is registered
       earlier (alphabetical file order), and this file is loaded second,
       **this version wins**.  If you want both, change ``_output_name``
       to ``'ret_clean'``.
    """
    panel = raw_tables["__panel__"]
    cleaned = panel["ret"].fillna(0)
    # Optional: clip extreme values
    cleaned = cleaned.clip(lower=-2.0, upper=2.0)
    return cleaned


clean_return.needs = {"crsp.sf": ["ret"]}
clean_return._output_name = "ret"   # ← alias: overwrites raw 'ret' column
clean_return._order = 5             # runs very early (cleaning phase)
