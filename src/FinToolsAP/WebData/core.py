"""
Logging Configuration
---------------------
This module creates a logger via ``logging.getLogger(__name__)``, which
resolves to ``FinToolsAP.WebData.core``.

By default, Python's ``logging`` module does **not** write to any file
or output stream unless explicitly configured by the application that
imports this module.  Without configuration, log messages at levels
below ``WARNING`` are silently discarded.

To enable log output, the consuming application must configure a
handler.  Examples:

    # Log to console (stderr)
    logging.basicConfig(level=logging.DEBUG)

    # Log to a file
    logging.basicConfig(
        filename="fintools.log",
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",

    # Configure only this module's logger
    logger = logging.getLogger("FinToolsAP.WebData.core")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler("webdata_core.log")
    logger.addHandler(handler)

Log levels used in this module:
    - ``DEBUG``   : SQL query strings for each WRDS table fetch.
    - ``INFO``    : Aggregated table needs summary.
    - ``WARNING`` : Characteristic functions returning unexpected types.
    - ``ERROR``   : Exceptions raised during characteristic computation
                    (with full traceback via ``exc_info=True``).
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
import re
import typing
from typing import Any, Dict, List, Optional, Union

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
        "crsp.sf":          "CRSP.MSF",
        "crsp.seall":       "CRSP.MSEALL",
        "crsp.si":          "CRSP.MSI",
        "crsp.link":        "CRSP.CCMXPF_LINKTABLE",
        "comp.fundq":       "COMP.FUNDQ",
        "comp.funda":       "COMP.FUNDA",
        "ibes.det_epsus":   "IBES.DET_EPSUS",
    },
    "D": {
        "crsp.sf":          "CRSP.DSF",
        "crsp.seall":       "CRSP.DSEALL",
        "crsp.si":          "CRSP.DSI",
        "crsp.link":        "CRSP.CCMXPF_LINKTABLE",
        "comp.fundq":       "COMP.FUNDQ",
        "comp.funda":       "COMP.FUNDA",
        "ibes.det_epsus":   "IBES.DET_EPSUS",
    },
}

# Default date column for each table alias
DATE_COL: dict[str, str] = {
    "crsp.sf":          "date",
    "crsp.seall":       "date",
    "crsp.si":          "date",
    "crsp.link":        None,      # link table has no single date column
    "comp.fundq":       "datadate",
    "comp.funda":       "datadate",
    "ibes.det_epsus":   "anndats",
}

# Identity columns always included in output
IDENTITY_COLS = ["ticker", "date", "permco"]

# Standard Compustat deduplication filter:  standard data format,
# domestic population, consolidated statements.
# NOTE: indfmt is intentionally omitted — 'INDL' would exclude firms
# classified under 'FS' (financial services format, e.g. GE with
# GE Capital).  Excluding financials is a sample-selection choice,
# not a data-quality filter.
_COMP_STD_FILTER = (
    "datafmt = 'STD' "
    "AND popsrc = 'D' AND consol = 'C'"
)


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
    numeric_ids: bool = False,
) -> str:
    """Build a simple ``SELECT … FROM … WHERE …`` query string.

    Parameters
    ----------
    numeric_ids : bool
        If ``True``, ID values are emitted without surrounding quotes so
        that WRDS numeric columns (``permco``, ``permno``) match without
        requiring an implicit cast.
    """
    col_str = ", ".join(columns)
    sql = f"SELECT {col_str} FROM {table_name}"

    clauses: list[str] = []
    if date_col is not None:
        clauses.append(f"{date_col} BETWEEN '{start_date}' AND '{end_date}'")
    if id_col is not None and ids is not None:
        if numeric_ids:
            vals = ", ".join(str(v) for v in ids)
        else:
            vals = ", ".join(f"'{v}'" for v in ids)
        clauses.append(f"{id_col} IN ({vals})")
    if predicates:
        clauses.append(predicates)

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    return sql


# ═══════════════════════════════════════════════════════════════════════════
# Raw column passthrough helpers
# ═══════════════════════════════════════════════════════════════════════════

# Shorthand table hints for raw column passthrough requests.
# Users MUST prefix a column name with one of these to target the
# appropriate WRDS table (e.g. ``chars=["compq.saleq", "crspsf.vol"]``).
_RAW_TABLE_HINTS: dict[str, str] = {
    "crspsf":  "crsp.sf",
    "crspse":  "crsp.seall",
    "crspsi":  "crsp.si",
    "compq":   "comp.fundq",
    "compa":   "comp.funda",
    "ibes":    "ibes.det_epsus",
}


# Regex for IBES forecast-period-indicator requests: ibes.fpi1, ibes.fpi3, etc.
_IBES_FPI_RE = re.compile(r"^ibes\.fpi(\d+)$", re.IGNORECASE)


def _is_ibes_fpi(name: str) -> int | None:
    """If *name* matches ``ibes.fpi<N>``, return *N* as an int; else ``None``."""
    m = _IBES_FPI_RE.match(name)
    return int(m.group(1)) if m else None


def _parse_raw_col(name: str) -> tuple[str, str]:
    """Parse a raw column request with a required table prefix.

    Returns ``(table_alias, column_name)``.

    Raises
    ------
    ValueError
        If *name* does not contain a recognised table prefix.

    Examples
    --------
    >>> _parse_raw_col("crspsf.prc")
    ('crsp.sf', 'prc')
    >>> _parse_raw_col("compq.saleq")
    ('comp.fundq', 'saleq')
    >>> _parse_raw_col("compa.at")
    ('comp.funda', 'at')
    >>> _parse_raw_col("crspse.shrcd")
    ('crsp.seall', 'shrcd')
    >>> _parse_raw_col("ibes.value")
    ('ibes.det_epsus', 'value')
    """
    if "." in name:
        prefix, col = name.split(".", 1)
        table = _RAW_TABLE_HINTS.get(prefix.lower())
        if table is not None:
            return table, col
    raise ValueError(
        f"Unrecognised characteristic or raw column request: {name!r}. "
        f"If requesting a raw WRDS column, prefix it with one of "
        f"{sorted(_RAW_TABLE_HINTS)} (e.g. 'crspsf.prc', 'compq.saleq', "
        f"'compa.at', 'crspse.shrcd', 'ibes.value'), or use 'ibes.fpiN' "
        f"for IBES forecast-period indicators (e.g. 'ibes.fpi1', 'ibes.fpi3')."
    )


def _make_raw_passthrough(output_name: str, table: str, col_name: str) -> Characteristic:
    """Create a :class:`Characteristic` that passes through a raw WRDS column."""

    def _passthrough(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
        panel = raw_tables["__panel__"]
        if col_name in panel.columns:
            return panel[col_name]
        return pd.Series(np.nan, index=panel.index)

    return Characteristic(
        name=output_name,
        func=_passthrough,
        dependencies={table: [col_name]},
        requires=[],
        description=f"Raw passthrough of '{col_name}' from {table}.",
        order=200,
    )


def _make_ibes_fpi_char(fpi: int) -> Characteristic:
    """Create a :class:`Characteristic` for an IBES fpi-specific consensus EPS estimate.

    The resulting characteristic filters ``raw_tables['ibes.det_epsus']``
    to the given ``fpi`` value, computes the mean analyst estimate per
    ``(date, cusip)``, and maps the result onto the CRSP panel.

    Parameters
    ----------
    fpi : int
        Forecast period indicator (e.g. 1 for one-year, 3 for three-year).

    Returns
    -------
    Characteristic
        With output name ``'eps_est_fpi{fpi}'``.
    """
    output_name = f"eps_est_fpi{fpi}"

    def _ibes_fpi_func(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
        panel = raw_tables["__panel__"]
        ibes = raw_tables.get("ibes.det_epsus")

        if ibes is None or ibes.empty or "cusip" not in panel.columns:
            return pd.Series(np.nan, index=panel.index)

        # Filter to the specific fpi value
        ibes_fpi = ibes[ibes["fpi"].astype(str) == str(fpi)]
        if ibes_fpi.empty:
            return pd.Series(np.nan, index=panel.index)

        # Consensus: mean analyst estimate per (date, cusip)
        consensus = (
            ibes_fpi.groupby(["date", "cusip"])[["value"]]
            .mean()
            .reset_index()
            .rename(columns={"value": output_name})
        )

        # Merge onto panel
        merged = panel[["date", "cusip"]].merge(
            consensus, on=["date", "cusip"], how="left",
        )
        return merged[output_name]

    return Characteristic(
        name=output_name,
        func=_ibes_fpi_func,
        dependencies={"ibes.det_epsus": ["cusip", "analys", "value", "anndats", "fpi"]},
        requires=[],
        description=f"Consensus (mean) EPS estimate from IBES with fpi={fpi}.",
        order=60,
    )


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
        tickers: list[str] | None = None,
        permcos: list[int | str] | None = None,
        permnos: list[int | str] | None = None,
        cusips: list[str] | None = None,
        gvkeys: list[str] | None = None,
        start_date: Any = None,
        end_date: Any = None,
        chars: list[str] | None = None,
        freq: str = "M",
        exchcd_filter: list[int] | None = [1, 2, 3],  # NYSE/AMEX/NASDAQ by default
        shrcd_filter: list[int] | None = [10, 11],  # Common shares by default
        ff_dataset: str | None = None,
    ) -> Union[pd.DataFrame, dict]:
        """Fetch WRDS data and compute requested characteristics.

        Parameters
        ----------
        tickers : list[str] or None
            Stock tickers (CRSP) to retrieve.
        permcos : list[int | str] or None
            CRSP permanent company identifiers.
        permnos : list[int | str] or None
            CRSP permanent security identifiers.
        cusips : list[str] or None
            CUSIP identifiers (matched via CRSP).
        gvkeys : list[str] or None
            Compustat Global Company Keys.  When provided the engine
            searches Compustat first and maps back to CRSP via the
            CCM link table.
        start_date, end_date : date-like
            Date range (inclusive).  Defaults to 1900-01-01 / today.
        chars : list[str] or None
            Characteristic names to compute, or raw WRDS column names.
            ``None`` → default set ``['prc', 'me', 'ret']``.
            Unregistered names are treated as raw column passthroughs
            and **must** be prefixed with a table hint:
            ``'crspsf.'`` (CRSP stock file), ``'crspse.'`` (CRSP
            events), ``'compq.'`` (Compustat quarterly), ``'compa.'``
            (Compustat annual), ``'crspsi.'`` (CRSP index), or
            ``'ibes.'`` (IBES).  If the column does not exist in the
            target table the WRDS query will raise an error.

            IBES forecast-period indicators can be requested via
            ``'ibes.fpi1'``, ``'ibes.fpi3'``, etc.  This fetches
            ``IBES.DET_EPSUS`` filtered to the specified fpi value
            and produces the characteristic ``eps_est_fpiN``.
        freq : {'M', 'D'}
            ``'M'`` for monthly, ``'D'`` for daily.
        exchcd_filter : list[int], optional
            Exchange code filter (e.g. ``[1, 2, 3]`` for NYSE/AMEX/NASDAQ).
        shrcd_filter : list[int], optional
            Share code filter (e.g. ``[10, 11]`` for common shares).
        ff_dataset : str, optional
            Name of a Ken French data library dataset to fetch via
            ``pandas_datareader`` (e.g. ``'F-F_Research_Data_Factors'``).
            **Cannot** be used simultaneously with WRDS identifiers
            (``tickers``, ``permcos``, etc.).  When provided, the
            return value is a ``dict[int, DataFrame]`` matching the
            ``pandas_datareader`` output format.

        Returns
        -------
        pandas.DataFrame or dict
            When using WRDS: panel with ``(ticker, date, permco)``
            identity columns plus one column per requested
            characteristic.
            When using ``ff_dataset``: a ``dict`` as returned by
            ``pandas_datareader.data.DataReader``.
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

        # ── Fama-French branch (mutually exclusive with WRDS) ────────
        if ff_dataset is not None:
            wrds_ids = [tickers, permcos, permnos, cusips, gvkeys]
            if any(v is not None for v in wrds_ids):
                raise ValueError(
                    "Cannot combine ff_dataset with WRDS identifiers "
                    "(tickers, permcos, permnos, cusips, gvkeys). "
                    "Fama-French data is an aggregate dataset and cannot "
                    "be filtered by individual securities."
                )
            try:
                import pandas_datareader.data as pdr
            except ImportError:
                raise ImportError(
                    "pandas_datareader is required for Fama-French data. "
                    "Install it with: pip install pandas-datareader"
                )
            logger.info("Fetching Fama-French dataset: %s", ff_dataset)
            ff_data: dict = pdr.DataReader(
                ff_dataset,
                "famafrench",
                start=start_date,
                end=end_date,
            )
            return ff_data

        if chars is None:
            chars = ["prc", "me", "ret"]

        if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars):
            raise TypeError("chars must be a list of strings.")

        # ── Detect IBES fpi requests (e.g. 'ibes.fpi1', 'ibes.fpi3') ────
        ibes_fpi_values: set[int] = set()
        ibes_fpi_chars: list[str] = []  # original names like 'ibes.fpi1'
        clean_chars: list[str] = []
        for c in chars:
            fpi_val = _is_ibes_fpi(c)
            if fpi_val is not None:
                ibes_fpi_values.add(fpi_val)
                ibes_fpi_chars.append(c)
            else:
                clean_chars.append(c)

        # Replace chars with the clean list (ibes.fpiN handled separately)
        chars = clean_chars

        # ── 1. Resolve chars against registry (+ auto-include deps) ────
        # Separate registered chars from raw passthrough requests.
        # Unregistered names are treated as raw WRDS columns and MUST
        # be prefixed with a table hint (e.g. "compq.saleq" → Compustat
        # quarterly, "crspsf.vol" → CRSP stock file, "ibes.value" → IBES).
        registered_names: list[str] = []
        raw_passthrough: list[tuple[str, str, str]] = []  # (orig, table, col)
        for c in chars:
            if c in self.registry:
                registered_names.append(c)
            else:
                table, col = _parse_raw_col(c)
                raw_passthrough.append((c, table, col))

        # Resolve registered characteristics with dependency expansion
        if registered_names:
            char_objects, _ = self.registry.resolve_with_deps(registered_names)
        else:
            char_objects = []

        # Create dynamic passthrough characteristics for raw column requests
        for orig_name, table, col in raw_passthrough:
            char_objects.append(_make_raw_passthrough(col, table, col))

        # Create dynamic IBES fpi characteristics
        for fpi_name in ibes_fpi_chars:
            fpi_val = _is_ibes_fpi(fpi_name)
            char_objects.append(_make_ibes_fpi_char(fpi_val))

        # Build the output-column name list (uses resolved column names)
        requested_names: list[str] = []
        for c in chars:
            if c in self.registry:
                requested_names.append(c)
            else:
                _, col = _parse_raw_col(c)
                requested_names.append(col)
        # Add IBES fpi output names
        for fpi_name in ibes_fpi_chars:
            fpi_val = _is_ibes_fpi(fpi_name)
            requested_names.append(f"eps_est_fpi{fpi_val}")

        # ── Collect IBES fpi values declared by characteristic functions ──
        for co in char_objects:
            func_fpi = getattr(co.func, "_ibes_fpi", None)
            if func_fpi:
                ibes_fpi_values.update(func_fpi)

        # ── 2. Aggregate .needs by table ─────────────────────────────────
        needs_by_table: dict[str, set[str]] = self.registry.aggregate_needs(
            char_objects
        )
        # Ensure IBES table is in needs if any fpi was requested
        if ibes_fpi_values:
            ibes_needs = needs_by_table.setdefault("ibes.det_epsus", set())
            ibes_needs |= {"cusip", "analys", "value", "anndats", "fpi"}
        logger.info("Aggregated needs: %s", needs_by_table)

        # ── 3. Fetch raw tables from WRDS ────────────────────────────────
        raw_tables: dict[str, pd.DataFrame] = self._fetch_raw_tables(
            needs_by_table=needs_by_table,
            tickers=tickers,
            permcos=permcos,
            permnos=permnos,
            cusips=cusips,
            gvkeys=gvkeys,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            exchcd_filter=exchcd_filter,
            shrcd_filter=shrcd_filter,
            ibes_fpi_values=ibes_fpi_values,
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
        # Store engine reference so characteristic functions (e.g. maxret)
        # can issue auxiliary queries when needed.
        raw_tables["__engine__"] = self
        panel: pd.DataFrame = raw_tables["__panel__"]
        panel = self._execute_chars(
            panel=panel,
            raw_tables=raw_tables,
            char_objects=char_objects,
            freq=freq,
        )

        # ── 6. Select output columns ────────────────────────────────────
        # Only include characteristics the user explicitly asked for, not
        # auto-included prerequisites (e.g. 'be' pulled in by 'bm').
        #
        # Dynamically extend identity columns to include whichever
        # identifier the user searched by (permno, cusip, gvkey) so the
        # search key always appears in the result.
        id_cols = list(IDENTITY_COLS)  # always: ticker, date, permco
        if permnos is not None and "permno" not in id_cols:
            id_cols.append("permno")
        if cusips is not None and "cusip" not in id_cols:
            id_cols.append("cusip")
        if gvkeys is not None and "gvkey" not in id_cols:
            id_cols.append("gvkey")

        output_cols = id_cols + requested_names
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
        permcos: list[int | str] | None,
        permnos: list[int | str] | None,
        cusips: list[str] | None,
        gvkeys: list[str] | None,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        freq: str,
        exchcd_filter: list[int] | None,
        shrcd_filter: list[int] | None,
        ibes_fpi_values: set[int] | None = None,
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
        if "comp.fundq" in tables_needed or "comp.funda" in tables_needed:
            tables_needed.add("crsp.link")
            tables_needed.add("crsp.seall")
            tables_needed.add("crsp.sf")

        # If IBES is needed, we also need CRSP SEALL for cusip-based merge
        if "ibes.det_epsus" in tables_needed:
            tables_needed.add("crsp.seall")

        # If user supplied gvkeys, pull in the link table so we can map
        # gvkeys → permcos for CRSP filtering
        if gvkeys is not None:
            tables_needed.add("crsp.link")
            tables_needed.add("crsp.seall")
            tables_needed.add("crsp.sf")

        # ── CCM Link Table (fetch early so gvkey→permco mapping is available)
        if "crsp.link" in tables_needed:
            link_cols = [
                "gvkey", "lpermco", "linktype", "linkprim",
                "linkdt", "linkenddt",
            ]
            link_pred = None
            if gvkeys is not None:
                quoted = ", ".join(f"'{v}'" for v in gvkeys)
                link_pred = f"gvkey IN ({quoted})"
            sql = _build_sql(
                table_name=table_map["crsp.link"],
                columns=link_cols,
                date_col=None,
                start_date=start_str,
                end_date=end_str,
                predicates=link_pred,
            )
            logger.debug("LINK SQL: %s", sql)
            raw_tables["crsp.link"] = self.wrds_conn.raw_sql(sql)

        # If user provided gvkeys, derive permcos from the link table
        _gvkey_permcos: list[str] | None = None
        if gvkeys is not None and "crsp.link" in raw_tables and not raw_tables["crsp.link"].empty:
            _gvkey_permcos = (
                raw_tables["crsp.link"]["lpermco"]
                .dropna()
                .astype(int)
                .astype(str)
                .unique()
                .tolist()
            )

        # ── CRSP Security-Event (SEALL) ─────────────────────────────────
        if "crsp.seall" in tables_needed:
            se_need_cols = needs_by_table.get("crsp.seall", set())
            se_cols = list(
                se_need_cols
                | {"date", "ticker", "comnam", "cusip",
                   "hsiccd", "permco", "permno", "shrcd", "exchcd"}
            )
            preds: list[str] = []
            if exchcd_filter:
                vals = ", ".join(str(x) for x in exchcd_filter)
                preds.append(f"exchcd IN ({vals})")
            if shrcd_filter:
                vals = ", ".join(str(x) for x in shrcd_filter)
                preds.append(f"shrcd IN ({vals})")

            # Determine the appropriate ID filter for the SEALL query.
            # Priority: tickers > permcos > permnos > cusips > gvkey-derived permcos
            id_col: str | None = None
            id_vals: list[str] | None = None
            _numeric_ids = False
            if tickers is not None:
                id_col, id_vals = "ticker", [str(t) for t in tickers]
            elif permcos is not None:
                id_col, id_vals = "permco", [str(p) for p in permcos]
                _numeric_ids = True
            elif permnos is not None:
                id_col, id_vals = "permno", [str(p) for p in permnos]
                _numeric_ids = True
            elif cusips is not None:
                id_col, id_vals = "cusip", [str(c) for c in cusips]
            elif _gvkey_permcos is not None:
                id_col, id_vals = "permco", _gvkey_permcos
                _numeric_ids = True

            pred_str = " AND ".join(preds) if preds else None
            sql = _build_sql(
                table_name=table_map["crsp.seall"],
                columns=se_cols,
                date_col="date",
                start_date=start_str,
                end_date=end_str,
                id_col=id_col,
                ids=id_vals,
                predicates=pred_str,
                numeric_ids=_numeric_ids,
            )
            logger.debug("SEALL SQL: %s", sql)
            raw_tables["crsp.seall"] = self.wrds_conn.raw_sql(sql)

        # ── CRSP Stock File (SF) ────────────────────────────────────────
        if "crsp.sf" in tables_needed:
            sf_cols_needed = needs_by_table.get("crsp.sf", set())
            sf_cols = list(
                sf_cols_needed
                | {"date", "permco", "cfacpr", "cfacshr"}
            )
            sf_permcos: list[str] | None = None
            if "crsp.seall" in raw_tables and not raw_tables["crsp.seall"].empty:
                sf_permcos = (
                    raw_tables["crsp.seall"]["permco"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

            # Fallback: if SEALL returned empty but the user explicitly
            # provided permcos or permnos, use them directly so the SF
            # query is never unfiltered.
            sf_id_col: str | None = None
            sf_id_vals: list[str] | None = None
            _sf_numeric = True

            if sf_permcos:
                sf_id_col = "permco"
                sf_id_vals = sf_permcos
            elif permcos is not None:
                sf_id_col = "permco"
                sf_id_vals = [str(p) for p in permcos]
            elif permnos is not None:
                sf_id_col = "permno"
                sf_id_vals = [str(p) for p in permnos]

            sql = _build_sql(
                table_name=table_map["crsp.sf"],
                columns=sf_cols,
                date_col="date",
                start_date=start_str,
                end_date=end_str,
                id_col=sf_id_col,
                ids=sf_id_vals,
                numeric_ids=_sf_numeric,
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

        # ── Compustat helper: gvkeys from link table ─────────────────────
        _link_gvkeys: list[str] | None = None
        if "crsp.link" in raw_tables and not raw_tables["crsp.link"].empty:
            _link_gvkeys = (
                raw_tables["crsp.link"]["gvkey"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        # ── Compustat Fundamentals Quarterly ─────────────────────────────
        if "comp.fundq" in tables_needed:
            comp_cols_needed = needs_by_table.get("comp.fundq", set())
            comp_cols = list(comp_cols_needed | {"gvkey", "datadate"})
            sql = _build_sql(
                table_name=table_map["comp.fundq"],
                columns=comp_cols,
                date_col="datadate",
                start_date=start_str,
                end_date=end_str,
                id_col="gvkey" if _link_gvkeys else None,
                ids=_link_gvkeys,
                predicates=_COMP_STD_FILTER,
            )
            logger.debug("COMPQ SQL: %s", sql)
            raw_tables["comp.fundq"] = self.wrds_conn.raw_sql(sql)

        # ── Compustat Fundamentals Annual ─────────────────────────────────
        if "comp.funda" in tables_needed:
            compa_cols_needed = needs_by_table.get("comp.funda", set())
            compa_cols = list(compa_cols_needed | {"gvkey", "datadate"})
            sql = _build_sql(
                table_name=table_map["comp.funda"],
                columns=compa_cols,
                date_col="datadate",
                start_date=start_str,
                end_date=end_str,
                id_col="gvkey" if _link_gvkeys else None,
                ids=_link_gvkeys,
                predicates=_COMP_STD_FILTER,
            )
            logger.debug("COMPA SQL: %s", sql)
            raw_tables["comp.funda"] = self.wrds_conn.raw_sql(sql)

        # ── IBES Detail EPS (DET_EPSUS) ─────────────────────────────────
        if "ibes.det_epsus" in tables_needed:
            ibes_cols = list(
                needs_by_table.get("ibes.det_epsus", set())
                | {"cusip", "analys", "value", "anndats", "fpi"}
            )
            # Build fpi predicate from requested values, default to fpi=1
            _fpi_vals = ibes_fpi_values if ibes_fpi_values else {1}
            fpi_quoted = ", ".join(f"'{v}'" for v in sorted(_fpi_vals))
            fpi_pred = f"fpi IN ({fpi_quoted})"
            sql = _build_sql(
                table_name=table_map["ibes.det_epsus"],
                columns=ibes_cols,
                date_col="anndats",
                start_date=start_str,
                end_date=end_str,
                predicates=fpi_pred,
            )
            logger.debug("IBES SQL: %s", sql)
            raw_tables["ibes.det_epsus"] = self.wrds_conn.raw_sql(sql)

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
                "hsiccd": "Int64", "permco": "Int64", "permno": "Int64",
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

            # Remove ±inf values from all numeric stock-file columns
            _numeric_sf_cols = [
                "prc", "shrout", "ret", "retx", "vol",
                "bidlo", "askhi", "cfacpr", "cfacshr",
            ]
            for col in _numeric_sf_cols:
                if col in sf.columns:
                    sf[col] = sf[col].replace([np.inf, -np.inf], np.nan)

            if "permco" in sf.columns:
                sf["permco"] = sf["permco"].astype("Int64")

            raw_tables["crsp.sf"] = sf

            # Merge onto panel by (date, permco)
            if not panel.empty and "permco" in panel.columns:
                panel = panel.merge(sf, on=["date", "permco"], how="left", suffixes=("", "_sf"))
            else:
                panel = sf.copy()

        # ── 4c. Clean & apply CCM Link Table ─────────────────────────────
        #  The link table maps CRSP permcos → Compustat gvkeys.  We clean
        #  it here and, if any Compustat table is present, merge the gvkey
        #  onto the panel so that steps 4d/4d-bis can do merge_asof joins.
        _any_comp = (
            ("comp.fundq" in raw_tables and not raw_tables["comp.fundq"].empty)
            or ("comp.funda" in raw_tables and not raw_tables["comp.funda"].empty)
        )

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

            # Apply link to panel if any Compustat table is needed
            if _any_comp and not panel.empty and "permco" in panel.columns:
                panel = panel.merge(
                    link, on="permco", how="left", suffixes=("", "_link")
                )
                # Keep only rows where date falls within link validity window
                mask = (
                    (panel["date"] >= panel["linkdt"])
                    & (panel["date"] <= panel["linkenddt"])
                )
                panel = panel[mask].copy()

        # ── 4d. Clean & merge Compustat Quarterly ────────────────────────
        if "comp.fundq" in raw_tables and not raw_tables["comp.fundq"].empty:
            comp = raw_tables["comp.fundq"].copy()
            comp["datadate"] = pd.to_datetime(comp["datadate"])
            comp["gvkey"] = comp["gvkey"].astype(str)
            raw_tables["comp.fundq"] = comp

            if not panel.empty and "gvkey" in panel.columns:
                comp_sorted = comp.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

                has_gvkey = panel["gvkey"].notna() & (panel["gvkey"] != "")
                panel_linked = panel[has_gvkey].copy()
                panel_unlinked = panel[~has_gvkey].copy()

                if not panel_linked.empty:
                    panel_linked = panel_linked.drop_duplicates(
                        subset=["gvkey", "date"], keep="first"
                    )
                    panel_linked["date"] = pd.to_datetime(panel_linked["date"])
                    comp_sorted["datadate"] = pd.to_datetime(comp_sorted["datadate"])
                    panel_linked = panel_linked.sort_values("date").reset_index(drop=True)
                    comp_sorted = comp_sorted.sort_values("datadate").reset_index(drop=True)
                    panel_linked = pd.merge_asof(
                        panel_linked,
                        comp_sorted,
                        left_on="date",
                        right_on="datadate",
                        by="gvkey",
                        direction="backward",
                        suffixes=("", "_compq"),
                    )

                panel = pd.concat([panel_linked, panel_unlinked], ignore_index=True)
                panel = panel.drop(columns=["datadate"], errors="ignore")

        # ── 4d-bis. Clean & merge Compustat Annual ───────────────────────
        if "comp.funda" in raw_tables and not raw_tables["comp.funda"].empty:
            compa = raw_tables["comp.funda"].copy()
            compa["datadate"] = pd.to_datetime(compa["datadate"])
            compa["gvkey"] = compa["gvkey"].astype(str)
            raw_tables["comp.funda"] = compa

            if not panel.empty and "gvkey" in panel.columns:
                compa_sorted = compa.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

                has_gvkey = panel["gvkey"].notna() & (panel["gvkey"] != "")
                panel_linked = panel[has_gvkey].copy()
                panel_unlinked = panel[~has_gvkey].copy()

                if not panel_linked.empty:
                    panel_linked = panel_linked.drop_duplicates(
                        subset=["gvkey", "date"], keep="first"
                    )
                    panel_linked["date"] = pd.to_datetime(panel_linked["date"])
                    compa_sorted["datadate"] = pd.to_datetime(compa_sorted["datadate"])
                    panel_linked = panel_linked.sort_values("date").reset_index(drop=True)
                    compa_sorted = compa_sorted.sort_values("datadate").reset_index(drop=True)
                    panel_linked = pd.merge_asof(
                        panel_linked,
                        compa_sorted,
                        left_on="date",
                        right_on="datadate",
                        by="gvkey",
                        direction="backward",
                        suffixes=("", "_compa"),
                    )

                panel = pd.concat([panel_linked, panel_unlinked], ignore_index=True)
                panel = panel.drop(columns=["datadate"], errors="ignore")

        # Clean up CCM link columns now that both Compustat merges are done
        if _any_comp:
            panel = panel.drop(
                columns=["linkdt", "linkenddt", "gvkey_compq", "gvkey_compa"],
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

        # ── 4f. Clean IBES ───────────────────────────────────────────────
        #  Only clean dates / cusips here.  Aggregation (e.g. consensus
        #  mean) is done inside the individual characteristic functions
        #  in ``definitions/ibes_chars.py`` so that different IBES-based
        #  characteristics can choose their own aggregation logic.
        if "ibes.det_epsus" in raw_tables and not raw_tables["ibes.det_epsus"].empty:
            ibes = raw_tables["ibes.det_epsus"].copy()
            ibes["anndats"] = pd.to_datetime(
                ibes["anndats"], format="%Y-%m-%d", errors="coerce"
            )
            ibes["anndats"] += pd.offsets.MonthEnd(0)
            ibes = ibes.dropna(subset=["anndats", "cusip"])
            ibes = ibes.rename(columns={"anndats": "date"})
            ibes["cusip"] = ibes["cusip"].str.strip()
            raw_tables["ibes.det_epsus"] = ibes

            # Ensure cusip on the panel is clean for downstream merges
            if not panel.empty and "cusip" in panel.columns:
                panel["cusip"] = panel["cusip"].astype(str).str.strip()

        # ── Store the merged panel ───────────────────────────────────────
        raw_tables["__panel__"] = panel
        return raw_tables

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Execute characteristic functions
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_column(series: pd.Series) -> pd.Series:
        """Normalise a column so that hidden ``np.nan`` and ``±inf`` become
        proper missing values.

        Pandas nullable float dtypes (e.g. ``Float64``) can store
        ``np.nan`` as a regular float value *distinct* from ``pd.NA``.
        ``.isna()`` / ``.notna()`` do not detect these hidden NaNs, so
        aggregations like ``.sum()`` treat them as real numbers —
        poisoning entire groups (NaN + anything = NaN).

        This helper:
        1. Replaces ``np.inf`` / ``-np.inf`` with ``np.nan``.
        2. If the dtype is a pandas nullable float (``Float64`` etc.),
           round-trips through numpy ``float64`` and back, which
           converts any hidden ``np.nan`` into a proper ``pd.NA``.
        """
        # Replace ±inf with NaN
        series = series.replace([np.inf, -np.inf], np.nan)

        # Round-trip nullable float to flush hidden np.nan → pd.NA
        if pd.api.types.is_float_dtype(series) or str(series.dtype).startswith("Float"):
            original_dtype = series.dtype
            series = series.astype("float64").astype(original_dtype)

        return series

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
                    panel[char.name] = self._sanitize_column(panel[char.name])
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        panel[col] = result[col].values
                        panel[col] = self._sanitize_column(panel[col])
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
