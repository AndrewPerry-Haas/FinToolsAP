"""
FinToolsAP.WebData.core
========================

The **engine** that orchestrates WRDS data fetching, table merging, and
characteristic computation.

Architecture
------------
::

    User calls getData(tickers, chars, ...)
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  1. Resolve chars → Characteristic list  │
    │  2. Aggregate .needs by table            │
    │  3. Fetch raw tables from WRDS           │
    │  4. ──── MERGE ZONE ────                 │
    │     User-defined table cleaning /        │
    │     joining lives HERE                   │
    │  5. Execute char functions in order       │
    │  6. Return final DataFrame               │
    └─────────────────────────────────────────┘

The engine is intentionally agnostic to the *content* of characteristic
functions.  All domain logic lives in ``definitions/*.py``.

Key design decisions
--------------------
* ``raw_tables`` is a ``dict[str, DataFrame]`` keyed by WRDS alias
  (e.g. ``'crsp.sf'``, ``'comp.fundq'``, ``'crsp.si'``).
* A special key ``'__panel__'`` holds the merged panel after the merge
  zone.  Ratio characteristics that need columns produced by earlier
  characteristics read from ``'__panel__'``.
* The merge zone is a clearly delimited method
  (:meth:`WebDataEngine._merge_raw_tables`) where all table joining,
  CCM link application, and date alignment takes place.
"""

from __future__ import annotations

import datetime
import logging
import typing
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import wrds as wrds_lib
except ImportError:
    wrds_lib = None  # type: ignore[assignment]

from .registry import REGISTRY, CharRegistry, Characteristic

logger = logging.getLogger(__name__)

# Pandas ≥2.2 renamed monthly offset alias from 'M' to 'ME'
_MONTH_END_RULE: str = "ME" if hasattr(pd.offsets, "MonthEnd") else "M"
try:
    pd.tseries.frequencies.to_offset(_MONTH_END_RULE)
except ValueError:
    _MONTH_END_RULE = "M"


# ═══════════════════════════════════════════════════════════════════════════
# Table alias → WRDS schema.table mapping
# ═══════════════════════════════════════════════════════════════════════════

# Keys used in .needs dicts          →  Actual WRDS table names
TABLE_MAP: dict[str, dict[str, str]] = {
    "M": {
        "crsp.sf":     "CRSP.MSF",
        "crsp.seall":  "CRSP.MSEALL",
        "crsp.si":     "CRSP.MSI",
        "crsp.link":   "CRSP.CCMXPF_LINKTABLE",
        "comp.fundq":  "COMP.FUNDQ",
    },
    "D": {
        "crsp.sf":     "CRSP.DSF",
        "crsp.seall":  "CRSP.DSEALL",
        "crsp.si":     "CRSP.DSI",
        "crsp.link":   "CRSP.CCMXPF_LINKTABLE",
        "comp.fundq":  "COMP.FUNDQ",
    },
}

# Default date column for each table alias
DATE_COL: dict[str, str] = {
    "crsp.sf":     "date",
    "crsp.seall":  "date",
    "crsp.si":     "date",
    "crsp.link":   None,      # link table has no single date column
    "comp.fundq":  "datadate",
}

# Identity columns always included in output
IDENTITY_COLS = ["ticker", "date", "permco"]


# ═══════════════════════════════════════════════════════════════════════════
# SQL builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_sql(
    table_name: str,
    columns: list[str],
    date_col: str | None,
    start_date: str,
    end_date: str,
    id_col: str | None = None,
    ids: list[str] | None = None,
    predicates: str | None = None,
) -> str:
    """Build a simple ``SELECT … FROM … WHERE …`` query string."""
    col_str = ", ".join(columns)
    sql = f"SELECT {col_str} FROM {table_name}"

    clauses: list[str] = []
    if date_col is not None:
        clauses.append(f"{date_col} BETWEEN '{start_date}' AND '{end_date}'")
    if id_col is not None and ids is not None:
        quoted = ", ".join(f"'{v}'" for v in ids)
        clauses.append(f"{id_col} IN ({quoted})")
    if predicates:
        clauses.append(predicates)

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    return sql


# ═══════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════

class WebDataEngine:
    """Core orchestrator for WRDS data retrieval and characteristic computation.

    Parameters
    ----------
    username : str
        WRDS username for authentication.
    registry : CharRegistry, optional
        Characteristic registry.  Defaults to the global ``REGISTRY``.
    """

    def __init__(
        self,
        username: str,
        registry: CharRegistry | None = None,
    ) -> None:
        self.username = username
        self.registry = registry or REGISTRY

        # ── WRDS connection ──────────────────────────────────────────────
        if wrds_lib is None or not hasattr(wrds_lib, "Connection"):
            raise ImportError(
                "The 'wrds' package is not available or is shadowed. "
                "Install it with: pip install wrds"
            )
        self.wrds_conn = wrds_lib.Connection(username=self.username)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def get_data(
        self,
        tickers: list[str] | None,
        start_date: Any = None,
        end_date: Any = None,
        chars: list[str] | None = None,
        freq: str = "M",
        exchcd_filter: list[int] | None = None,
        shrcd_filter: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch WRDS data and compute requested characteristics.

        Parameters
        ----------
        tickers : list[str] or None
            Stock tickers to retrieve.  ``None`` triggers a full-universe pull.
        start_date, end_date : date-like
            Date range (inclusive).  Defaults to 1900-01-01 / today.
        chars : list[str] or None
            Characteristic names to compute.  ``None`` → default set.
        freq : {'M', 'D'}
            ``'M'`` for monthly, ``'D'`` for daily.
        exchcd_filter : list[int], optional
            Exchange code filter (e.g. ``[1, 2, 3]`` for NYSE/AMEX/NASDAQ).
        shrcd_filter : list[int], optional
            Share code filter (e.g. ``[10, 11]`` for common shares).

        Returns
        -------
        pandas.DataFrame
            Panel with ``(ticker, date, permco)`` identity columns plus one
            column per requested characteristic.
        """
        # ── Validate inputs ──────────────────────────────────────────────
        start_date = pd.to_datetime(
            start_date or datetime.datetime(1900, 1, 1), errors="raise"
        )
        end_date = pd.to_datetime(
            end_date or datetime.datetime.now(), errors="raise"
        )
        if start_date > end_date:
            raise ValueError("start_date must be before end_date.")
        if freq not in ("M", "D"):
            raise ValueError("freq must be 'M' or 'D'.")

        if chars is None:
            chars = ["prc", "me", "ret"]

        if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars):
            raise TypeError("chars must be a list of strings.")

        # ── 1. Resolve chars against registry ────────────────────────────
        char_objects: list[Characteristic] = self.registry.resolve(chars)

        # ── 2. Aggregate .needs by table ─────────────────────────────────
        needs_by_table: dict[str, set[str]] = self.registry.aggregate_needs(
            char_objects
        )
        logger.info("Aggregated needs: %s", needs_by_table)

        # ── 3. Fetch raw tables from WRDS ────────────────────────────────
        raw_tables: dict[str, pd.DataFrame] = self._fetch_raw_tables(
            needs_by_table=needs_by_table,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            exchcd_filter=exchcd_filter,
            shrcd_filter=shrcd_filter,
        )

        # ══════════════════════════════════════════════════════════════════
        #
        #                      ┌──────────────┐
        #                      │  MERGE ZONE  │
        #                      └──────────────┘
        #
        #  This is the clearly-defined area where raw WRDS tables are
        #  cleaned, joined, and assembled into a single panel DataFrame.
        #
        #  raw_tables is {'wrds_table_alias': DataFrame, ...}
        #  After this call, raw_tables['__panel__'] contains the merged
        #  panel keyed by (permco, date) with identity columns attached.
        #
        # ══════════════════════════════════════════════════════════════════
        raw_tables = self._merge_raw_tables(
            raw_tables=raw_tables,
            freq=freq,
        )

        # ── 5. Execute characteristic functions in dependency order ──────
        panel: pd.DataFrame = raw_tables["__panel__"]
        panel = self._execute_chars(
            panel=panel,
            raw_tables=raw_tables,
            char_objects=char_objects,
            freq=freq,
        )

        # ── 6. Select output columns ────────────────────────────────────
        output_cols = IDENTITY_COLS + [c.name for c in char_objects]
        # Keep only columns that actually exist (some chars may not produce
        # output if upstream data was missing)
        output_cols = [c for c in output_cols if c in panel.columns]
        panel = panel[output_cols].copy()
        panel = panel.reset_index(drop=True)

        return panel

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Fetch raw tables
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_raw_tables(
        self,
        needs_by_table: dict[str, set[str]],
        tickers: list[str] | None,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        freq: str,
        exchcd_filter: list[int] | None,
        shrcd_filter: list[int] | None,
    ) -> dict[str, pd.DataFrame]:
        """Issue one SQL query per required WRDS table and return the results.

        Returns
        -------
        dict[str, DataFrame]
            Keyed by table alias (e.g. ``'crsp.sf'``).
        """
        table_map = TABLE_MAP[freq]
        raw_tables: dict[str, pd.DataFrame] = {}
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Determine which tables we actually need
        tables_needed = set(needs_by_table.keys())

        # If any CRSP stock-file table is needed, we also need the identity
        # table (SEALL) to resolve tickers → permcos
        if "crsp.sf" in tables_needed:
            tables_needed.add("crsp.seall")

        # If any Compustat table is needed, we also need the CCM link table
        if "comp.fundq" in tables_needed:
            tables_needed.add("crsp.link")
            tables_needed.add("crsp.seall")
            tables_needed.add("crsp.sf")

        # ── CRSP Security-Event (SEALL) ─────────────────────────────────
        if "crsp.seall" in tables_needed:
            se_cols = list({"date", "ticker", "comnam", "cusip",
                            "hsiccd", "permco", "shrcd", "exchcd"})
            preds = []
            if exchcd_filter:
                vals = ", ".join(str(x) for x in exchcd_filter)
                preds.append(f"exchcd IN ({vals})")
            if shrcd_filter:
                vals = ", ".join(str(x) for x in shrcd_filter)
                preds.append(f"shrcd IN ({vals})")
            pred_str = " AND ".join(preds) if preds else None

            sql = _build_sql(
                table_name=table_map["crsp.seall"],
                columns=se_cols,
                date_col="date",
                start_date=start_str,
                end_date=end_str,
                id_col="ticker" if tickers else None,
                ids=tickers,
                predicates=pred_str,
            )
            logger.debug("SEALL SQL: %s", sql)
            raw_tables["crsp.seall"] = self.wrds_conn.raw_sql(sql)

        # ── CRSP Stock File (SF) ────────────────────────────────────────
        if "crsp.sf" in tables_needed:
            # We need permcos from SEALL to scope the SF query
            sf_cols_needed = needs_by_table.get("crsp.sf", set())
            # Always fetch identity + adjustment factors alongside requested cols
            sf_cols = list(
                sf_cols_needed
                | {"date", "permco", "cfacpr", "cfacshr"}
            )
            permcos: list[str] | None = None
            if "crsp.seall" in raw_tables and not raw_tables["crsp.seall"].empty:
                permcos = (
                    raw_tables["crsp.seall"]["permco"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            sql = _build_sql(
                table_name=table_map["crsp.sf"],
                columns=sf_cols,
                date_col="date",
                start_date=start_str,
                end_date=end_str,
                id_col="permco" if permcos else None,
                ids=permcos,
            )
            logger.debug("SF SQL: %s", sql)
            raw_tables["crsp.sf"] = self.wrds_conn.raw_sql(sql)

        # ── CRSP Index (SI) ─────────────────────────────────────────────
        if "crsp.si" in tables_needed:
            si_cols = list(needs_by_table.get("crsp.si", set()) | {"date"})
            sql = _build_sql(
                table_name=table_map["crsp.si"],
                columns=si_cols,
                date_col="date",
                start_date=start_str,
                end_date=end_str,
            )
            logger.debug("SI SQL: %s", sql)
            raw_tables["crsp.si"] = self.wrds_conn.raw_sql(sql)

        # ── CCM Link Table ──────────────────────────────────────────────
        if "crsp.link" in tables_needed:
            link_cols = [
                "gvkey", "lpermco", "linktype", "linkprim",
                "linkdt", "linkenddt",
            ]
            sql = _build_sql(
                table_name=table_map["crsp.link"],
                columns=link_cols,
                date_col=None,
                start_date=start_str,
                end_date=end_str,
            )
            logger.debug("LINK SQL: %s", sql)
            raw_tables["crsp.link"] = self.wrds_conn.raw_sql(sql)

        # ── Compustat Fundamentals Quarterly ─────────────────────────────
        if "comp.fundq" in tables_needed:
            comp_cols_needed = needs_by_table.get("comp.fundq", set())
            comp_cols = list(comp_cols_needed | {"gvkey", "datadate"})
            # Scope to gvkeys from link table if available
            gvkeys: list[str] | None = None
            if "crsp.link" in raw_tables and not raw_tables["crsp.link"].empty:
                gvkeys = (
                    raw_tables["crsp.link"]["gvkey"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            sql = _build_sql(
                table_name=table_map["comp.fundq"],
                columns=comp_cols,
                date_col="datadate",
                start_date=start_str,
                end_date=end_str,
                id_col="gvkey" if gvkeys else None,
                ids=gvkeys,
            )
            logger.debug("COMP SQL: %s", sql)
            raw_tables["comp.fundq"] = self.wrds_conn.raw_sql(sql)

        return raw_tables

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: MERGE ZONE
    # ──────────────────────────────────────────────────────────────────────

    def _merge_raw_tables(
        self,
        raw_tables: dict[str, pd.DataFrame],
        freq: str,
    ) -> dict[str, pd.DataFrame]:
        """
        ╔══════════════════════════════════════════════════════════════════╗
        ║                         MERGE ZONE                             ║
        ║                                                                ║
        ║  This method is the SINGLE PLACE where raw WRDS tables are     ║
        ║  cleaned, joined, and assembled into a unified panel.          ║
        ║                                                                ║
        ║  INPUT:  raw_tables = {'crsp.sf': df, 'crsp.seall': df, ...}  ║
        ║  OUTPUT: raw_tables with added key '__panel__'                 ║
        ║                                                                ║
        ║  Customize this method for your specific cleaning / joining    ║
        ║  requirements. The characteristic functions downstream only    ║
        ║  see the result.                                               ║
        ╚══════════════════════════════════════════════════════════════════╝
        """

        panel = pd.DataFrame()

        # ── 4a. Clean CRSP Security-Event (identity) ─────────────────────
        if "crsp.seall" in raw_tables and not raw_tables["crsp.seall"].empty:
            se = raw_tables["crsp.seall"].copy()
            se["date"] = pd.to_datetime(se["date"])
            se = se.sort_values(["ticker", "date"])
            se = se.drop_duplicates(subset=["ticker", "date"])

            # Resample to fill gaps within each ticker's time series
            resample_rule = "D" if freq == "D" else _MONTH_END_RULE
            se = se.set_index("date")
            # Select only value columns for resample to avoid
            # FutureWarning about operating on grouping columns
            value_cols = [c for c in se.columns if c != "ticker"]
            se = (
                se.groupby("ticker")[value_cols]
                .resample(resample_rule)
                .ffill()
                .reset_index()
            )
            if freq == "M":
                se["date"] = se["date"] + pd.offsets.MonthEnd(0)

            # Type coercion
            type_map = {
                "ticker": str, "comnam": str, "cusip": str,
                "hsiccd": "Int64", "permco": "Int64",
                "shrcd": "Int64", "exchcd": "Int64",
            }
            for col, dtype in type_map.items():
                if col in se.columns:
                    se[col] = se[col].astype(dtype)

            raw_tables["crsp.seall"] = se
            panel = se.copy()

        # ── 4b. Clean & merge CRSP Stock File ────────────────────────────
        if "crsp.sf" in raw_tables and not raw_tables["crsp.sf"].empty:
            sf = raw_tables["crsp.sf"].copy()
            sf["date"] = pd.to_datetime(sf["date"])
            if freq == "M":
                sf["date"] = sf["date"] + pd.offsets.MonthEnd(0)
            sf = sf.drop_duplicates(subset=["permco", "date"])

            # Type coercion for available columns
            sf_types = {
                "permco": "Int64", "prc": float, "shrout": float,
                "ret": float, "retx": float, "vol": float,
                "bidlo": float, "askhi": float,
                "cfacpr": float, "cfacshr": float,
            }
            for col, dtype in sf_types.items():
                if col in sf.columns:
                    sf[col] = pd.to_numeric(sf[col], errors="coerce")

            if "permco" in sf.columns:
                sf["permco"] = sf["permco"].astype("Int64")

            raw_tables["crsp.sf"] = sf

            # Merge onto panel by (date, permco)
            if not panel.empty and "permco" in panel.columns:
                panel = panel.merge(sf, on=["date", "permco"], how="left", suffixes=("", "_sf"))
            else:
                panel = sf.copy()

        # ── 4c. Clean & apply CCM Link Table ─────────────────────────────
        if "crsp.link" in raw_tables and not raw_tables["crsp.link"].empty:
            link = raw_tables["crsp.link"].copy()
            link = link.rename(columns={"lpermco": "permco"})

            # Keep only standard link types
            link = link[link["linktype"].str.startswith("L", na=False)]
            link = link[link["linkprim"].isin(["C", "P"])]

            # Prefer 'P' links
            link["_prim_order"] = np.where(link["linkprim"] == "P", 0, 1)
            link = link.sort_values(["permco", "_prim_order", "linkdt"])

            link = link.dropna(subset=["permco"])
            link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
            link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2200-01-01"))
            link["linkdt"] = pd.to_datetime(link["linkdt"], errors="coerce")

            link["permco"] = link["permco"].astype("Int64")
            link["gvkey"] = link["gvkey"].astype(str)

            link = link.drop_duplicates(subset=["permco", "gvkey", "linkdt", "linkenddt"])
            link = link[["gvkey", "permco", "linkdt", "linkenddt"]]
            link = link.drop(columns=["_prim_order"], errors="ignore")

            raw_tables["crsp.link"] = link

        # ── 4d. Clean & merge Compustat ──────────────────────────────────
        if "comp.fundq" in raw_tables and not raw_tables["comp.fundq"].empty:
            comp = raw_tables["comp.fundq"].copy()
            comp["datadate"] = pd.to_datetime(comp["datadate"])
            comp["gvkey"] = comp["gvkey"].astype(str)

            raw_tables["comp.fundq"] = comp

            # Join Compustat to panel via CCM link
            if (
                "crsp.link" in raw_tables
                and not raw_tables["crsp.link"].empty
                and not panel.empty
                and "permco" in panel.columns
            ):
                link = raw_tables["crsp.link"]
                # Merge link onto panel
                panel = panel.merge(
                    link, on="permco", how="left", suffixes=("", "_link")
                )
                # Keep only rows where date falls within link validity window
                mask = (
                    (panel["date"] >= panel["linkdt"])
                    & (panel["date"] <= panel["linkenddt"])
                )
                panel = panel[mask].copy()

                # Now merge Compustat onto the linked panel
                # Use the most recent Compustat observation ≤ panel date
                # merge_asof requires both sides sorted on the merge key
                comp_sorted = comp.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

                # Drop rows with missing gvkey before asof merge, then
                # rejoin to preserve them (they just won't have Comp data)
                has_gvkey = panel["gvkey"].notna() & (panel["gvkey"] != "")
                panel_linked = panel[has_gvkey].copy()
                panel_unlinked = panel[~has_gvkey].copy()

                if not panel_linked.empty:
                    # Deduplicate to keep one row per (gvkey, date)
                    panel_linked = panel_linked.drop_duplicates(
                        subset=["gvkey", "date"], keep="first"
                    )
                    # Ensure proper types for the asof join
                    panel_linked["date"] = pd.to_datetime(panel_linked["date"])
                    comp_sorted["datadate"] = pd.to_datetime(comp_sorted["datadate"])
                    # merge_asof requires the left_on column to be globally
                    # sorted (even when using `by`).  Sort by date.
                    panel_linked = panel_linked.sort_values("date").reset_index(drop=True)
                    comp_sorted = comp_sorted.sort_values("datadate").reset_index(drop=True)
                    panel_linked = pd.merge_asof(
                        panel_linked,
                        comp_sorted,
                        left_on="date",
                        right_on="datadate",
                        by="gvkey",
                        direction="backward",
                        suffixes=("", "_comp"),
                    )

                panel = pd.concat([panel_linked, panel_unlinked], ignore_index=True)

                # Clean up link columns
                panel = panel.drop(
                    columns=["linkdt", "linkenddt", "datadate", "gvkey_comp"],
                    errors="ignore",
                )

        # ── 4e. Merge index data ─────────────────────────────────────────
        if "crsp.si" in raw_tables and not raw_tables["crsp.si"].empty:
            idx = raw_tables["crsp.si"].copy()
            idx["date"] = pd.to_datetime(idx["date"])
            if freq == "M":
                idx["date"] = idx["date"] + pd.offsets.MonthEnd(0)
            raw_tables["crsp.si"] = idx

            if not panel.empty and "date" in panel.columns:
                panel = panel.merge(idx, on="date", how="left", suffixes=("", "_idx"))
            else:
                panel = idx.copy()

        # ── Store the merged panel ───────────────────────────────────────
        raw_tables["__panel__"] = panel
        return raw_tables

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Execute characteristic functions
    # ──────────────────────────────────────────────────────────────────────

    def _execute_chars(
        self,
        panel: pd.DataFrame,
        raw_tables: dict[str, pd.DataFrame],
        char_objects: list[Characteristic],
        freq: str,
    ) -> pd.DataFrame:
        """Run each characteristic function and assign results to the panel.

        Functions are executed in ``order`` priority (already sorted by
        ``registry.resolve``).  After each function, the panel is updated
        so that later functions can read earlier results via
        ``raw_tables['__panel__']``.
        """
        for char in char_objects:
            try:
                # Keep __panel__ in sync so ratio chars can read predecessors
                raw_tables["__panel__"] = panel
                result = char.func(raw_tables, freq)

                if isinstance(result, pd.Series):
                    panel[char.name] = result.values
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        panel[col] = result[col].values
                else:
                    logger.warning(
                        "Characteristic %r returned %s instead of Series/DataFrame; skipping.",
                        char.name,
                        type(result).__name__,
                    )
            except Exception:
                logger.error(
                    "Error computing characteristic %r", char.name, exc_info=True
                )
                panel[char.name] = np.nan

        return panel

    # ──────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ──────────────────────────────────────────────────────────────────────

    def available_chars(self) -> list[str]:
        """Return all characteristic names the registry knows about."""
        return self.registry.available()

    def describe_chars(self, name: str | None = None) -> dict[str, str]:
        """Return ``{name: description}`` for one or all characteristics."""
        return self.registry.describe(name)

    # ──────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the WRDS connection."""
        try:
            if self.wrds_conn is not None:
                self.wrds_conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        chars = ", ".join(self.available_chars()[:10])
        if len(self.available_chars()) > 10:
            chars += ", ..."
        return (
            f"WebDataEngine(username={self.username!r}, "
            f"chars=[{chars}] ({len(self.registry)} total))"
        )
