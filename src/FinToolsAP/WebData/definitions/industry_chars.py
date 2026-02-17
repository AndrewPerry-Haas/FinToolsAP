"""
FinToolsAP.WebData.definitions.industry_chars
==============================================

Fama-French industry classification characteristics.

Downloads SIC code range definitions from Kenneth French's data library
at runtime and maps each stock's four-digit SIC code (``hsiccd``) to its
Fama-French industry group.

Classifications
---------------
ind5, ind10, ind12, ind17, ind30, ind38, ind48, ind49

All are order-1 characteristics with no frequency restriction (they
work identically at monthly and daily frequency).

Variables Used
--------------
* ``hsiccd`` — four-digit historical SIC code from ``CRSP.MSEALL`` /
  ``CRSP.DSEALL``.
"""

from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests


# ═══════════════════════════════════════════════════════════════════════════
# SIC code range downloader  (cached for the lifetime of the process)
# ═══════════════════════════════════════════════════════════════════════════

_SIC_RANGE_RE = re.compile(r"^\d{4}-\d{4}$")


@lru_cache(maxsize=16)
def _download_ff_sic_ranges(
    n_industries: int,
) -> List[Tuple[int, int, str]]:
    """Download and parse Fama-French SIC code ranges.

    Returns
    -------
    list of (sic_start, sic_end, industry_abbrev)
    """
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        f"ftp/Siccodes{n_industries}.zip"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw = zf.read(txt_name).decode("latin1")

    ranges: List[Tuple[int, int, str]] = []
    curr_abbrev: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        parts = [p.strip() for p in stripped.split(" ", 1)]
        if parts[0] == "":
            continue
        if _SIC_RANGE_RE.match(parts[0]):
            sic_start, sic_end = map(int, parts[0].split("-"))
            ranges.append((sic_start, sic_end, curr_abbrev or "Other"))
        else:
            header = [p.strip() for p in stripped.split(" ", 2)]
            curr_abbrev = header[1] if len(header) > 1 else header[0]

    return ranges


def _map_sic_to_industry(hsiccd: pd.Series, n_industries: int) -> pd.Series:
    """Map a Series of 4-digit SIC codes to FF industry abbreviations."""
    ranges = _download_ff_sic_ranges(n_industries)
    sic = hsiccd.astype(float)

    result = pd.Series("Other", index=hsiccd.index, dtype=object)
    for sic_start, sic_end, abbrev in ranges:
        mask = (sic >= sic_start) & (sic <= sic_end)
        result = result.where(~mask, abbrev)

    # Where hsiccd is NaN → mark classification as NaN
    result = result.where(sic.notna(), other=np.nan)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Characteristic functions
# ═══════════════════════════════════════════════════════════════════════════

def ind5(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 5-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 5)

ind5.needs = {"crsp.seall": ["hsiccd"]}
ind5._output_name = "ind5"
ind5._order = 1


def ind10(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 10-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 10)

ind10.needs = {"crsp.seall": ["hsiccd"]}
ind10._output_name = "ind10"
ind10._order = 1


def ind12(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 12-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 12)

ind12.needs = {"crsp.seall": ["hsiccd"]}
ind12._output_name = "ind12"
ind12._order = 1


def ind17(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 17-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 17)

ind17.needs = {"crsp.seall": ["hsiccd"]}
ind17._output_name = "ind17"
ind17._order = 1


def ind30(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 30-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 30)

ind30.needs = {"crsp.seall": ["hsiccd"]}
ind30._output_name = "ind30"
ind30._order = 1


def ind38(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 38-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 38)

ind38.needs = {"crsp.seall": ["hsiccd"]}
ind38._output_name = "ind38"
ind38._order = 1


def ind48(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 48-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 48)

ind48.needs = {"crsp.seall": ["hsiccd"]}
ind48._output_name = "ind48"
ind48._order = 1


def ind49(raw_tables: dict[str, pd.DataFrame], freq: str) -> pd.Series:
    """Fama-French 49-industry classification."""
    panel = raw_tables["__panel__"]
    return _map_sic_to_industry(panel["hsiccd"], 49)

ind49.needs = {"crsp.seall": ["hsiccd"]}
ind49._output_name = "ind49"
ind49._order = 1
