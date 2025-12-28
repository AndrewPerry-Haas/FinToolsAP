from __future__ import annotations

import datetime
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .builders import BuildContext, REGISTRY, BuilderError
from .catalog import (
    CatalogError,
    FeatureRef,
    FeatureSpec,
    LinkRef,
    TableColumnsRef,
    WrdsCatalogBundle,
    load_bundle,
    load_default_bundle,
)
from .planner import ExplainPlan, PlanError, explain_plan
from .plugin import load_entrypoint_builders


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    catalog_path: str | Path | None = None
    links_path: str | Path | None = None
    features_path: str | Path | None = None
    plugin_group: str = "fintoolsap_wrds.builders"


class WrdsEngine:
    """TOML-driven WRDS execution engine.

    The engine is fetcher-agnostic: it calls the provided loader callables.
    `FinToolsAP.WebData.WebData` wires these to its existing `_load_*` methods.
    """

    def __init__(
        self,
        *,
        fetch_se: typing.Callable[..., pd.DataFrame],
        fetch_sf: typing.Callable[..., pd.DataFrame],
        fetch_index: typing.Callable[..., pd.DataFrame],
        fetch_link: typing.Callable[..., pd.DataFrame],
        fetch_comp: typing.Callable[..., pd.DataFrame],
        clean_inputs: typing.Callable[..., tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]] | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self._fetch_se = fetch_se
        self._fetch_sf = fetch_sf
        self._fetch_index = fetch_index
        self._fetch_link = fetch_link
        self._fetch_comp = fetch_comp
        self._clean_inputs = clean_inputs
        self._config = config or EngineConfig()

        self._bundle: WrdsCatalogBundle | None = None

        # simple in-instance caches
        self._table_cache: dict[str, pd.DataFrame] = {}
        self._feature_cache: dict[str, pd.Series] = {}

    # --- public APIs ---
    def explain_plan(
        self,
        requested_features: list[str],
        *,
        include_identity: bool = False,
    ) -> ExplainPlan:
        bundle = self._load_catalogs()
        feats = list(requested_features)
        if include_identity:
            for f in ("ticker", "date", "permco"):
                if f not in feats and f in bundle.features:
                    feats.insert(0, f)
        return explain_plan(feats, tables=bundle.tables, links=bundle.links, features=bundle.features)

    def get(
        self,
        *,
        features: list[str],
        tickers: list[str] | None,
        freq: str,
        start_date: typing.Any = None,
        end_date: typing.Any = None,
        exchcd_filter: list[int] | None = None,
        shrcd_filter: list[int] | None = None,
        include_identity: bool = False,
    ) -> pd.DataFrame:
        if freq not in ("M", "D"):
            raise ValueError("freq must be 'M' or 'D'")

        start_ts, end_ts = self._normalize_dates(start_date, end_date, freq)
        start_sql = "'" + start_ts.strftime("%Y-%m-%d") + "'"
        end_sql = "'" + end_ts.strftime("%Y-%m-%d") + "'"

        bundle = self._load_catalogs()

        requested = list(features)
        if include_identity:
            for f in ("ticker", "date", "permco"):
                if f not in requested and f in bundle.features:
                    requested.insert(0, f)

        plan = explain_plan(requested, tables=bundle.tables, links=bundle.links, features=bundle.features)

        # load plugins (built-ins are registered on import) and validate builders
        load_entrypoint_builders(group=self._config.plugin_group, registry=REGISTRY, allow_override=False)
        for feat, builder in plan.builder_sequence:
            try:
                REGISTRY.get(builder)
            except BuilderError as e:
                raise EngineError(f"Missing builder '{builder}' required for feature '{feat}'.") from e

        # fetch required tables with column projection
        se_df = None
        sf_df = None
        idx_df = None
        link_df = None
        comp_df = None

        # SE may be needed either for output features OR just to map ticker->permco.
        needs_se_for_output = "crsp_se" in plan.fetch_plan
        needs_sf = "crsp_sf" in plan.fetch_plan
        needs_idx = "crsp_index" in plan.fetch_plan
        needs_comp = "comp_fundq" in plan.fetch_plan
        needs_link = "ccm_link" in plan.fetch_plan or ("crsp_comp" in plan.join_plan)

        se_cols_for_output = list(plan.fetch_plan.get("crsp_se", ()))
        # ensure mapping columns if we need them
        se_cols_for_mapping = ["date", "ticker", "permco"]

        if needs_se_for_output:
            se_cols = list(plan.fetch_plan["crsp_se"])
            se_df = self._fetch_se(
                tickers,
                start_sql,
                end_sql,
                freq,
                select_cols=se_cols,
                exchcd_filter=exchcd_filter,
                shrcd_filter=shrcd_filter,
            )

        # If SF is requested, we need permcos; derive them from SE even if SE isn't requested for output.
        if needs_sf and se_df is None:
            se_df = self._fetch_se(
                tickers,
                start_sql,
                end_sql,
                freq,
                select_cols=sorted(set(se_cols_for_mapping)),
                exchcd_filter=exchcd_filter,
                shrcd_filter=shrcd_filter,
            )

        permcos: list[str] | None = None
        if needs_sf:
            if se_df is None:
                raise EngineError(
                    "Cannot fetch CRSP SF without a permco universe. Provide tickers (so SE can map to permco) or request only SE features."
                )

            permcos = [str(x) for x in se_df.get("permco", pd.Series([], dtype="Int64")).dropna().unique()]
            if len(permcos) == 0:
                # empty universe
                return pd.DataFrame(columns=list(requested))

            sf_cols = list(plan.fetch_plan["crsp_sf"])
            sf_df = self._fetch_sf(permcos or [], start_sql, end_sql, freq, select_cols=sf_cols)

        if needs_idx:
            # bound index range to the actually fetched SF when present
            if sf_df is not None and not sf_df.empty and "date" in sf_df.columns:
                smin = pd.to_datetime(sf_df["date"]).min().strftime("%Y-%m-%d")
                smax = pd.to_datetime(sf_df["date"]).max().strftime("%Y-%m-%d")
                idx_start_sql = f"'{smin}'"
                idx_end_sql = f"'{smax}'"
            else:
                idx_start_sql, idx_end_sql = start_sql, end_sql

            idx_df = self._fetch_index(idx_start_sql, idx_end_sql, freq)

        if needs_link:
            link_df = self._fetch_link()

        if needs_comp:
            # comp requires link resolution; we fetch comp after applying link bounds
            # We defer until after base panel is created.
            pass

        # optional cleaning hook (WebData has an existing _clean_inputs)
        if self._clean_inputs is not None:
            se_df, sf_df, link_df, comp_df, idx_df = self._clean_inputs(se_df, sf_df, link_df, comp_df, idx_df, freq)
        else:
            se_df, sf_df, link_df, comp_df, idx_df = self._default_clean(se_df, sf_df, link_df, comp_df, idx_df, freq)

        # Build base panel. Only merge SE+SF when SE columns are required for requested features.
        base = self._build_base_panel(se_df=se_df, sf_df=sf_df, freq=freq, merge_se_sf=needs_se_for_output)

        # Apply Compustat join if needed
        if "crsp_comp" in plan.join_plan:
            if link_df is None:
                raise EngineError("Join plan requires 'crsp_comp' but CCM link table was not fetched")

            base = self._apply_ccm_comp_join(
                base=base,
                link_df=link_df,
                start_sql=start_sql,
                end_sql=end_sql,
                comp_cols=list(plan.fetch_plan.get("comp_fundq", ())),
                freq=freq,
            )

        # Apply index join if needed
        if idx_df is not None and not idx_df.empty:
            idx_df = idx_df.copy()
            idx_df["date"] = pd.to_datetime(idx_df["date"])
            if freq == "M":
                idx_df["date"] = idx_df["date"] + pd.tseries.offsets.MonthEnd(0)
            base = base.merge(idx_df, how="left", on=["date"])

        # Standard base index
        if "permco" not in base.columns or "date" not in base.columns:
            raise EngineError("Base panel missing required columns 'permco' and 'date'")
        base = base.drop_duplicates(subset=["permco", "date"]).sort_values(["permco", "date"]).reset_index(drop=True)
        base_index = pd.MultiIndex.from_frame(base[["permco", "date"]], names=["permco", "date"])

        # Build features in dependency order
        self._feature_cache = {}
        for feat in plan.expanded_features:
            self._materialize_feature(feat, bundle=bundle, base=base, base_index=base_index, freq=freq)

        # Assemble output
        # Build output frame.
        # `permco`/`date` are represented by the base index; don't create them as columns
        # before reset_index() or pandas will error on duplicate column insertion.
        out = pd.DataFrame(index=base_index)
        for f in requested:
            if f in ("permco", "date"):
                continue
            out[f] = self._feature_cache[f]

        out = out.reset_index()
        return out[list(requested)]

    # --- internals ---
    def _load_catalogs(self) -> WrdsCatalogBundle:
        if self._bundle is not None:
            return self._bundle

        # env overrides
        catalog_path = self._config.catalog_path or os.getenv("WRDS_CATALOG_PATH")
        links_path = self._config.links_path or os.getenv("WRDS_LINKS_PATH")
        features_path = self._config.features_path or os.getenv("WRDS_FEATURES_PATH")

        if catalog_path or links_path or features_path:
            if not (catalog_path and links_path and features_path):
                raise EngineError(
                    "If overriding catalogs via env/args, you must provide WRDS_CATALOG_PATH, WRDS_LINKS_PATH, and WRDS_FEATURES_PATH."
                )
            self._bundle = load_bundle(
                catalog_path=catalog_path,
                links_path=links_path,
                features_path=features_path,
            )
        else:
            self._bundle = load_default_bundle()

        return self._bundle

    def _normalize_dates(self, start_date: typing.Any, end_date: typing.Any, freq: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = pd.to_datetime(start_date or datetime.datetime(1900, 1, 1), errors="raise")
        end_ts = pd.to_datetime(end_date or datetime.datetime.now(), errors="raise")
        if start_ts > end_ts:
            raise ValueError("start_date must be before end_date")

        if freq == "M":
            # clip to prior month-end
            end_ts = (end_ts - pd.tseries.offsets.MonthBegin(1)) + pd.tseries.offsets.MonthEnd(0)
        return start_ts, end_ts

    def _default_clean(
        self,
        se_df: pd.DataFrame | None,
        sf_df: pd.DataFrame | None,
        link_df: pd.DataFrame | None,
        comp_df: pd.DataFrame | None,
        idx_df: pd.DataFrame | None,
        freq: str,
    ):
        # Minimal subset of WebData._clean_inputs
        if se_df is not None and not se_df.empty and "date" in se_df.columns and "ticker" in se_df.columns:
            se_df = se_df.copy()
            se_df["date"] = pd.to_datetime(se_df["date"])
            se_df = se_df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])

        if sf_df is not None and not sf_df.empty and "date" in sf_df.columns:
            sf_df = sf_df.copy()
            sf_df["date"] = pd.to_datetime(sf_df["date"])
            if freq == "M":
                sf_df["date"] = sf_df["date"] + pd.tseries.offsets.MonthEnd(0)
            if "permco" in sf_df.columns:
                sf_df = sf_df.dropna(subset=["permco"])

        if link_df is not None and not link_df.empty:
            link_df = link_df.copy()
            # normalize column name to match legacy merge expectations
            if "lpermco" in link_df.columns and "permco" not in link_df.columns:
                link_df = link_df.rename(columns={"lpermco": "permco"})
            if "linkdt" in link_df.columns:
                link_df["linkdt"] = pd.to_datetime(link_df["linkdt"], errors="coerce")
            if "linkenddt" in link_df.columns:
                link_df["linkenddt"] = pd.to_datetime(link_df["linkenddt"], errors="coerce")

        if comp_df is not None and not comp_df.empty:
            comp_df = comp_df.copy()
            if "datadate" in comp_df.columns:
                comp_df = comp_df.rename(columns={"datadate": "date"})
            comp_df["date"] = pd.to_datetime(comp_df["date"], errors="coerce")

        if idx_df is not None and not idx_df.empty and "date" in idx_df.columns:
            idx_df = idx_df.copy()
            idx_df["date"] = pd.to_datetime(idx_df["date"])
            if freq == "M":
                idx_df["date"] = idx_df["date"] + pd.tseries.offsets.MonthEnd(0)

        return se_df, sf_df, link_df, comp_df, idx_df

    def _build_base_panel(self, *, se_df: pd.DataFrame | None, sf_df: pd.DataFrame | None, freq: str, merge_se_sf: bool) -> pd.DataFrame:
        if sf_df is None and se_df is None:
            raise EngineError("No base tables fetched; cannot build panel")

        if sf_df is None:
            base = se_df.copy()  # type: ignore[union-attr]
            base["date"] = pd.to_datetime(base["date"])
            if freq == "M":
                base["date"] = base["date"] + pd.tseries.offsets.MonthEnd(0)
            # ensure permco/date exist
            if "permco" not in base.columns:
                raise EngineError("SE table did not include permco")
            base = base.dropna(subset=["permco"])
            return base

        base_sf = sf_df.copy()
        base_sf["date"] = pd.to_datetime(base_sf["date"])
        if freq == "M":
            base_sf["date"] = base_sf["date"] + pd.tseries.offsets.MonthEnd(0)

        if se_df is None or not merge_se_sf:
            return base_sf

        base_se = se_df.copy()
        base_se["date"] = pd.to_datetime(base_se["date"])
        if freq == "M":
            base_se["date"] = base_se["date"] + pd.tseries.offsets.MonthEnd(0)

        if "permco" not in base_se.columns:
            raise EngineError("SE table did not include permco")

        merged = base_sf.merge(base_se, how="inner", on=["date", "permco"])
        return merged

    def _apply_ccm_comp_join(
        self,
        *,
        base: pd.DataFrame,
        link_df: pd.DataFrame,
        start_sql: str,
        end_sql: str,
        comp_cols: list[str],
        freq: str,
    ) -> pd.DataFrame:
        link_df = link_df.copy()
        if "lpermco" in link_df.columns and "permco" not in link_df.columns:
            link_df = link_df.rename(columns={"lpermco": "permco"})

        # filter and prefer primary links when possible
        if "linkprim" in link_df.columns:
            link_df["_linkprim_order"] = (link_df["linkprim"].astype(str) != "P").astype(int)
            link_df = link_df.sort_values(["permco", "_linkprim_order", "linkdt"], kind="mergesort")

        # bounds types
        if "linkdt" in link_df.columns:
            link_df["linkdt"] = pd.to_datetime(link_df["linkdt"], errors="coerce")
        if "linkenddt" in link_df.columns:
            link_df["linkenddt"] = pd.to_datetime(link_df["linkenddt"], errors="coerce")

        # Merge base with link table, then apply date validity
        merged = base.merge(link_df, how="inner", on=["permco"])
        if "linkdt" in merged.columns and "linkenddt" in merged.columns:
            merged = merged[(merged["date"] >= merged["linkdt"]) & (merged["date"] <= merged["linkenddt"])]

        if "gvkey" not in merged.columns:
            raise EngineError("CCM link join did not produce gvkey")

        gvkeys = [str(x) for x in merged["gvkey"].dropna().unique()]
        if len(gvkeys) == 0:
            return merged

        # Fetch Compustat now that gvkeys are known
        # Ensure gvkey/date always present
        cols = set(comp_cols)
        cols.update({"gvkey", "datadate"})
        comp_df = self._fetch_comp(gvkeys, start_sql, end_sql, select_cols=sorted(cols))
        if comp_df is None or comp_df.empty:
            return merged

        comp_df = comp_df.copy()
        # normalize date column name
        if "datadate" in comp_df.columns:
            comp_df = comp_df.rename(columns={"datadate": "date"})
        comp_df["date"] = pd.to_datetime(comp_df["date"], errors="coerce")

        # align to quarter-end to mimic current behavior
        comp_df["date"] = comp_df["date"] + pd.tseries.offsets.QuarterEnd(0)

        # Merge and forward-fill by gvkey
        merged = merged.merge(comp_df, how="left", on=["gvkey", "date"])
        comp_value_cols = [c for c in comp_df.columns if c not in ("gvkey", "date")]
        if comp_value_cols:
            merged = merged.sort_values(["gvkey", "date"])
            merged[comp_value_cols] = merged.groupby("gvkey", group_keys=False)[comp_value_cols].ffill()

        # Collapse duplicates to one row per permco/date, preferring P links first
        merged = merged.sort_values(["permco", "date"])
        merged = merged.drop_duplicates(subset=["permco", "date"], keep="first")
        return merged

    def _materialize_feature(
        self,
        feat: str,
        *,
        bundle: WrdsCatalogBundle,
        base: pd.DataFrame,
        base_index: pd.MultiIndex,
        freq: str,
    ) -> None:
        if feat in self._feature_cache:
            return

        spec = bundle.features.get(feat)
        if spec is None:
            raise EngineError(f"Unknown feature '{feat}'")

        # raw: take from merged base
        if spec.kind == "raw":
            if spec.columns is None or len(spec.columns) != 1:
                raise EngineError(f"Raw feature '{feat}' must declare exactly one column")
            col = spec.columns[0]
            if col not in base.columns:
                # Some raw features are from a table that was not merged into base.
                raise EngineError(f"Column '{col}' for raw feature '{feat}' is not available on the base panel")
            s = pd.Series(base[col].values, index=base_index)
            self._feature_cache[feat] = s
            return

        # derived
        if not spec.builder:
            raise EngineError(f"Derived feature '{feat}' missing builder")

        # ensure feature deps are materialized
        inputs: dict[str, object] = {}
        for inp in spec.inputs:
            if isinstance(inp, FeatureRef):
                self._materialize_feature(inp.feature, bundle=bundle, base=base, base_index=base_index, freq=freq)
                inputs[inp.feature] = self._feature_cache[inp.feature]
            elif isinstance(inp, TableColumnsRef):
                # For split_adjust, pass an SF dataframe with keys+requested columns
                if spec.builder == "split_adjust" and inp.table == "crsp_sf":
                    cols = ["permco", "date"] + list(inp.columns)
                    cols = [c for c in cols if c in base.columns]
                    inputs["crsp_sf"] = base[cols].copy()
                else:
                    # otherwise, pass each requested column as a Series
                    for c in inp.columns:
                        if c not in base.columns:
                            raise EngineError(f"Required input column '{c}' for feature '{feat}' not found on base panel")
                        inputs[c] = pd.Series(base[c].values, index=base_index)
            elif isinstance(inp, LinkRef):
                # join requirements are handled by the plan/executor
                continue

        ctx = BuildContext(feature=feat, freq=freq, base_index=base_index)
        fn = REGISTRY.get(spec.builder)
        out = fn(inputs, ctx)
        if not isinstance(out, pd.Series):
            raise EngineError(f"Builder '{spec.builder}' for feature '{feat}' must return a pandas Series")
        out = out.reindex(base_index)
        self._feature_cache[feat] = out
