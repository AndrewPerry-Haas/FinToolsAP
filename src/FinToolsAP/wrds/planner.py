from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .catalog import (
    CatalogError,
    FeatureRef,
    FeatureSpec,
    LinkRef,
    LinkSpec,
    TableColumnsRef,
    TableSpec,
)


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class ExplainPlan:
    requested_features: tuple[str, ...]
    expanded_features: tuple[str, ...]
    fetch_plan: Mapping[str, tuple[str, ...]]  # table -> columns
    join_plan: tuple[str, ...]  # link names
    builder_sequence: tuple[tuple[str, str], ...]  # (feature, builder)

    def as_dict(self) -> dict:
        return {
            "requested_features": list(self.requested_features),
            "expanded_features": list(self.expanded_features),
            "fetch_plan": {k: list(v) for k, v in self.fetch_plan.items()},
            "join_plan": list(self.join_plan),
            "builder_sequence": [
                {"feature": f, "builder": b} for (f, b) in self.builder_sequence
            ],
        }


def resolve_feature_dag(
    requested_features: Sequence[str],
    *,
    feature_catalog: Mapping[str, FeatureSpec],
) -> list[str]:
    """Return topologically-sorted expanded feature list.

    Raises PlanError on missing features or cycles.
    """

    missing = [f for f in requested_features if f not in feature_catalog]
    if missing:
        raise PlanError(f"Unknown feature(s) requested: {missing}")

    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def deps(feat: FeatureSpec) -> list[str]:
        if feat.kind != "derived":
            return []
        out: list[str] = []
        for inp in feat.inputs:
            if isinstance(inp, FeatureRef):
                out.append(inp.feature)
        return out

    def dfs(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise PlanError(f"Cyclic feature dependency detected at '{name}'")
        visiting.add(name)

        feat = feature_catalog[name]
        for d in sorted(deps(feat)):
            if d not in feature_catalog:
                raise PlanError(f"Feature '{name}' depends on missing feature '{d}'")
            dfs(d)

        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for f in requested_features:
        dfs(f)

    return order


def compile_fetch_plan(
    expanded_features: Sequence[str],
    *,
    tables: Mapping[str, TableSpec],
    links: Mapping[str, LinkSpec],
    features: Mapping[str, FeatureSpec],
) -> dict[str, set[str]]:
    """Return minimal required columns per table."""

    fetch: dict[str, set[str]] = {}

    def add_cols(table_name: str, cols: Iterable[str]) -> None:
        if table_name not in tables:
            raise PlanError(f"Unknown table '{table_name}'")
        fetch.setdefault(table_name, set()).update(cols)
        # always include keys for stable indexing/merging
        fetch[table_name].update(tables[table_name].keys)

    for feat_name in expanded_features:
        spec = features[feat_name]
        if spec.kind == "raw":
            if spec.table is None:
                raise PlanError(f"Raw feature '{feat_name}' missing table")
            add_cols(spec.table, spec.columns)
            continue

        # derived inputs
        for inp in spec.inputs:
            if isinstance(inp, TableColumnsRef):
                add_cols(inp.table, inp.columns)
            elif isinstance(inp, LinkRef):
                link = links.get(inp.link)
                if link is None:
                    raise PlanError(f"Feature '{feat_name}' references unknown link '{inp.link}'")

                # Through link requires link table columns for joins + validity
                if link.through is not None:
                    through_table = link.through
                    if through_table is None:
                        raise PlanError(f"Link '{link.name}' missing through table")

                    # join columns
                    if link.left_to_through:
                        add_cols(through_table, link.left_to_through.values())
                    if link.through_to_right:
                        add_cols(through_table, link.through_to_right.keys())

                    # validity columns
                    if link.validity is not None:
                        add_cols(through_table, [link.validity.start_col, link.validity.end_col])

                    # prefer columns must be present
                    for col in (link.prefer or {}).keys():
                        add_cols(through_table, [col])

                else:
                    # direct link uses join keys on both tables
                    add_cols(link.left_table, link.keys.keys())
                    add_cols(link.right_table, link.keys.values())

    return fetch


def compile_join_plan(
    expanded_features: Sequence[str],
    *,
    links: Mapping[str, LinkSpec],
    features: Mapping[str, FeatureSpec],
) -> list[str]:
    needed: set[str] = set()
    for feat_name in expanded_features:
        spec = features[feat_name]
        if spec.kind != "derived":
            continue
        for inp in spec.inputs:
            if isinstance(inp, LinkRef):
                needed.add(inp.link)

    # deterministic order
    return [name for name in sorted(needed) if name in links]


def builder_sequence(
    expanded_features: Sequence[str],
    *,
    features: Mapping[str, FeatureSpec],
) -> list[tuple[str, str]]:
    seq: list[tuple[str, str]] = []
    for feat_name in expanded_features:
        spec = features[feat_name]
        if spec.kind == "derived":
            if not spec.builder:
                raise PlanError(f"Derived feature '{feat_name}' missing builder")
            seq.append((feat_name, spec.builder))
    return seq


def explain_plan(
    requested_features: Sequence[str],
    *,
    tables: Mapping[str, TableSpec],
    links: Mapping[str, LinkSpec],
    features: Mapping[str, FeatureSpec],
) -> ExplainPlan:
    expanded = resolve_feature_dag(requested_features, feature_catalog=features)
    fetch = compile_fetch_plan(expanded, tables=tables, links=links, features=features)
    joins = compile_join_plan(expanded, links=links, features=features)
    builders = builder_sequence(expanded, features=features)

    fetch_sorted = {t: tuple(sorted(cols)) for t, cols in sorted(fetch.items())}
    return ExplainPlan(
        requested_features=tuple(requested_features),
        expanded_features=tuple(expanded),
        fetch_plan=fetch_sorted,
        join_plan=tuple(joins),
        builder_sequence=tuple(builders),
    )
