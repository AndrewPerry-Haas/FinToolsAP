from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd


BuilderFn = Callable[[Dict[str, object], "BuildContext"], object]


class BuilderError(ValueError):
    pass


class BuilderRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, BuilderFn] = {}

    def register(self, name: str, fn: BuilderFn, *, override: bool = False) -> None:
        if not isinstance(name, str) or name.strip() == "":
            raise BuilderError("Builder name must be a non-empty string")
        if not callable(fn):
            raise BuilderError(f"Builder '{name}' must be callable")
        if name in self._builders and not override:
            raise BuilderError(
                f"Builder '{name}' is already registered; pass override=True to replace it."
            )
        self._builders[name] = fn

    def get(self, name: str) -> BuilderFn:
        try:
            return self._builders[name]
        except KeyError as e:
            raise BuilderError(f"Unknown builder '{name}'") from e

    def list_builders(self) -> list[str]:
        return sorted(self._builders.keys())


REGISTRY = BuilderRegistry()


def register_builder(name: str, *, override: bool = False):
    """Decorator to register a builder in the global registry."""

    def deco(fn: BuilderFn) -> BuilderFn:
        REGISTRY.register(name, fn, override=override)
        return fn

    return deco


@dataclass(frozen=True)
class BuildContext:
    feature: str
    freq: str
    base_index: pd.MultiIndex


def _as_series(x: object, *, feature: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0]
    raise BuilderError(f"Builder input for '{feature}' must be a Series (or 1-col DataFrame)")


def _ensure_index(s: pd.Series, ctx: BuildContext) -> pd.Series:
    if not s.index.equals(ctx.base_index):
        # allow reindex if compatible
        try:
            return s.reindex(ctx.base_index)
        except Exception as e:  # pragma: no cover
            raise BuilderError(f"Builder produced misaligned index for '{ctx.feature}': {e}") from e
    return s


@register_builder("split_adjust")
def split_adjust(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    """Split-adjust a CRSP SF column.

    Engine passes one DataFrame under key 'crsp_sf' containing required raw cols and
    the base keys ('permco','date') as columns.

    Returns a Series for the requested feature (ctx.feature).
    """
    df = inputs.get("crsp_sf")
    if not isinstance(df, pd.DataFrame):
        raise BuilderError("split_adjust expects inputs['crsp_sf'] as a DataFrame")

    feat = ctx.feature
    if feat not in {"prc", "shrout", "bidlo", "askhi"}:
        raise BuilderError(f"split_adjust cannot build feature '{feat}'")

    if "permco" not in df.columns or "date" not in df.columns:
        raise BuilderError("split_adjust requires 'permco' and 'date' columns")

    # construct series aligned to base index
    tmp = df.set_index(["permco", "date"]).sort_index()

    if feat in {"prc", "bidlo", "askhi"}:
        if feat not in tmp.columns or "cfacpr" not in tmp.columns:
            raise BuilderError(f"split_adjust('{feat}') requires columns '{feat}' and 'cfacpr'")
        s = tmp[feat].astype(float).abs() / tmp["cfacpr"].astype(float)
        return _ensure_index(s, ctx)

    # shrout
    if "shrout" not in tmp.columns or "cfacshr" not in tmp.columns:
        raise BuilderError("split_adjust('shrout') requires columns 'shrout' and 'cfacshr'")
    s = tmp["shrout"].astype(float) * tmp["cfacshr"].astype(float) / 1e3
    return _ensure_index(s, ctx)


@register_builder("build_me")
def build_me(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    prc = _ensure_index(_as_series(inputs.get("prc"), feature="prc"), ctx)
    shrout = _ensure_index(_as_series(inputs.get("shrout"), feature="shrout"), ctx)
    return prc * shrout


@register_builder("build_div")
def build_div(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    ret = _ensure_index(_as_series(inputs.get("ret"), feature="ret"), ctx).fillna(0.0)
    retx = _ensure_index(_as_series(inputs.get("retx"), feature="retx"), ctx).fillna(0.0)
    prc = _ensure_index(_as_series(inputs.get("prc"), feature="prc"), ctx)

    # IMPORTANT: shift must be within permco
    prc_prev = prc.groupby(level=0).shift(1)
    div = (ret - retx) * prc_prev
    return div.fillna(0.0)


@register_builder("build_div_12m_sum")
def build_div_12m_sum(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    div = _ensure_index(_as_series(inputs.get("div"), feature="div"), ctx)

    if ctx.freq not in ("M", "D"):
        raise BuilderError("freq must be 'M' or 'D'")

    window = 12 if ctx.freq == "M" else 252
    min_periods = 7 if ctx.freq == "M" else 147

    out = div.groupby(level=0).transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).sum()
    )
    return out


@register_builder("build_dp")
def build_dp(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    div_sum = _ensure_index(_as_series(inputs.get("div_12m_sum"), feature="div_12m_sum"), ctx)
    prc = _ensure_index(_as_series(inputs.get("prc"), feature="prc"), ctx)
    return pd.Series(np.where(prc != 0, div_sum / prc, np.nan), index=ctx.base_index)


@register_builder("build_dps")
def build_dps(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    div_sum = _ensure_index(_as_series(inputs.get("div_12m_sum"), feature="div_12m_sum"), ctx)
    shrout = _ensure_index(_as_series(inputs.get("shrout"), feature="shrout"), ctx)
    return pd.Series(np.where(shrout != 0, div_sum / shrout, np.nan), index=ctx.base_index)


@register_builder("build_pps")
def build_pps(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    prc = _ensure_index(_as_series(inputs.get("prc"), feature="prc"), ctx)
    shrout = _ensure_index(_as_series(inputs.get("shrout"), feature="shrout"), ctx)
    return pd.Series(np.where(shrout == 0, np.nan, prc / shrout), index=ctx.base_index)


@register_builder("build_be")
def build_be(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    """Compustat book equity.

    Standard (simplified) BE definition:
      BE = SEQ + TXDITC - PSTK
    where PSTK uses PSTKRQ if present else PSTKQ else 0.

    Engine supplies aligned Series after link+ffill.
    """
    seqq = _ensure_index(_as_series(inputs.get("seqq"), feature="seqq"), ctx)
    txditcq = _ensure_index(_as_series(inputs.get("txditcq"), feature="txditcq"), ctx).fillna(0.0)
    pstkrq = inputs.get("pstkrq")
    pstkq = inputs.get("pstkq")

    pstk = None
    if isinstance(pstkrq, pd.Series):
        pstk = _ensure_index(pstkrq, ctx)
    if pstk is None and isinstance(pstkq, pd.Series):
        pstk = _ensure_index(pstkq, ctx)
    if pstk is None:
        pstk = pd.Series(0.0, index=ctx.base_index)
    else:
        pstk = pstk.fillna(0.0)

    be = seqq.astype(float) + txditcq.astype(float) - pstk.astype(float)
    return be


@register_builder("build_earn")
def build_earn(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    ibq = _ensure_index(_as_series(inputs.get("ibq"), feature="ibq"), ctx)
    return ibq.astype(float)


@register_builder("build_earn_ann")
def build_earn_ann(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    """Annualize quarterly earnings.

    For now: `earn_ann = 4 * earn` (avoids rolling-sum artifacts when FUNDQ is
    forward-filled onto monthly/daily panels).
    """
    earn = _ensure_index(_as_series(inputs.get("earn"), feature="earn"), ctx)
    return 4.0 * earn.astype(float)


@register_builder("build_bm")
def build_bm(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    be = _ensure_index(_as_series(inputs.get("be"), feature="be"), ctx)
    me = _ensure_index(_as_series(inputs.get("me"), feature="me"), ctx)
    return pd.Series(np.where(me != 0, be / me, np.nan), index=ctx.base_index)


@register_builder("build_bps")
def build_bps(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    be = _ensure_index(_as_series(inputs.get("be"), feature="be"), ctx)
    shrout = _ensure_index(_as_series(inputs.get("shrout"), feature="shrout"), ctx)
    return pd.Series(np.where(shrout != 0, be / shrout, np.nan), index=ctx.base_index)


@register_builder("build_ep")
def build_ep(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    earn_ann = _ensure_index(_as_series(inputs.get("earn_ann"), feature="earn_ann"), ctx)
    me = _ensure_index(_as_series(inputs.get("me"), feature="me"), ctx)
    return pd.Series(np.where(me != 0, earn_ann / me, np.nan), index=ctx.base_index)


@register_builder("build_eps")
def build_eps(inputs: Dict[str, object], ctx: BuildContext) -> pd.Series:
    earn_ann = _ensure_index(_as_series(inputs.get("earn_ann"), feature="earn_ann"), ctx)
    shrout = _ensure_index(_as_series(inputs.get("shrout"), feature="shrout"), ctx)
    return pd.Series(np.where(shrout != 0, earn_ann / shrout, np.nan), index=ctx.base_index)
