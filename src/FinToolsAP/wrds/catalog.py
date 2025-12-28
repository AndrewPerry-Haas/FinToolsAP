from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import importlib.resources

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


class CatalogError(ValueError):
    """Raised when WRDS TOML catalogs are invalid."""


@dataclass(frozen=True)
class TableSpec:
    name: str
    schema: str
    keys: tuple[str, ...]
    date_col: Optional[str] = None
    id_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkValiditySpec:
    start_col: str
    end_col: str
    panel_date_col: str


@dataclass(frozen=True)
class LinkSpec:
    name: str
    left_table: str
    right_table: str
    # simple direct join: left_col -> right_col
    keys: Mapping[str, str] = field(default_factory=dict)

    # optional 3-table join
    through: Optional[str] = None
    left_to_through: Optional[Mapping[str, str]] = None
    through_to_right: Optional[Mapping[str, str]] = None

    validity: Optional[LinkValiditySpec] = None
    filters: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # table -> predicates
    prefer: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureRef:
    feature: str


@dataclass(frozen=True)
class TableColumnsRef:
    table: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class LinkRef:
    link: str


FeatureInputSpec = Union[FeatureRef, TableColumnsRef, LinkRef]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: Literal["raw", "derived"]

    # raw
    table: Optional[str] = None
    columns: tuple[str, ...] = ()

    # derived
    builder: Optional[str] = None
    inputs: tuple[FeatureInputSpec, ...] = ()


@dataclass(frozen=True)
class WrdsCatalogBundle:
    tables: Mapping[str, TableSpec]
    links: Mapping[str, LinkSpec]
    features: Mapping[str, FeatureSpec]


def _load_toml_dict(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CatalogError(f"TOML path does not exist: {p}")
    try:
        with p.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:  # pragma: no cover
        raise CatalogError(f"Failed to parse TOML at {p}: {e}") from e
    if not isinstance(data, dict):
        raise CatalogError(f"Invalid TOML root object in {p}; expected a TOML table.")
    return data


def _load_packaged_toml(filename: str) -> dict[str, Any]:
    """Load a TOML dict from packaged data (works in wheels/zip)."""
    try:
        content = importlib.resources.files("FinToolsAP.wrds").joinpath("data").joinpath(filename).read_bytes()
    except Exception as e:  # pragma: no cover
        raise CatalogError(f"Failed to load packaged TOML '{filename}': {e}") from e
    try:
        data = tomllib.loads(content.decode("utf-8"))
    except Exception as e:  # pragma: no cover
        raise CatalogError(f"Failed to parse packaged TOML '{filename}': {e}") from e
    if not isinstance(data, dict):
        raise CatalogError(f"Invalid TOML root object in packaged '{filename}'; expected a TOML table.")
    _validate_version(data, f"package:data/{filename}")
    return data


def load_default_bundle() -> WrdsCatalogBundle:
    """Load the default catalogs shipped with the package."""
    catalog = _load_packaged_toml("wrds_catalog.toml")
    links = _load_packaged_toml("wrds_links.toml")
    features = _load_packaged_toml("wrds_features.toml")

    tables = load_table_catalog_from_dict(catalog, source="package:data/wrds_catalog.toml")
    link_graph = load_link_graph_from_dict(links, source="package:data/wrds_links.toml")
    feat_catalog = load_feature_catalog_from_dict(features, source="package:data/wrds_features.toml")
    validate_catalog_bundle(tables=tables, links=link_graph, features=feat_catalog)
    return WrdsCatalogBundle(tables=tables, links=link_graph, features=feat_catalog)


def load_table_catalog_from_dict(data: Mapping[str, Any], *, source: str) -> dict[str, TableSpec]:
    _validate_version(data, source)
    tables = data.get("tables")
    if not isinstance(tables, dict) or len(tables) == 0:
        raise CatalogError(f"{source}: missing or empty [tables] section")

    res: dict[str, TableSpec] = {}
    for table_name, spec in tables.items():
        if not isinstance(spec, dict):
            raise CatalogError(f"{source}: tables.{table_name} must be a table")
        schema = _req_str(source, f"tables.{table_name}.schema", spec.get("schema"))
        name = _req_str(source, f"tables.{table_name}.name", spec.get("name"))
        keys = _req_str_list(source, f"tables.{table_name}.keys", spec.get("keys"))
        date_col = _opt_str(source, f"tables.{table_name}.date_col", spec.get("date_col"))
        id_cols = tuple(_opt_str_list(source, f"tables.{table_name}.id_cols", spec.get("id_cols")) or [])

        res[table_name] = TableSpec(
            name=name,
            schema=schema,
            keys=tuple(keys),
            date_col=date_col,
            id_cols=id_cols,
        )
    return res


def load_link_graph_from_dict(data: Mapping[str, Any], *, source: str) -> dict[str, LinkSpec]:
    _validate_version(data, source)
    links = data.get("links")
    if not isinstance(links, dict) or len(links) == 0:
        raise CatalogError(f"{source}: missing or empty [links] section")

    res: dict[str, LinkSpec] = {}
    for link_name, spec in links.items():
        if not isinstance(spec, dict):
            raise CatalogError(f"{source}: links.{link_name} must be a table")

        left_table = _req_str(source, f"links.{link_name}.left_table", spec.get("left_table"))
        right_table = _req_str(source, f"links.{link_name}.right_table", spec.get("right_table"))

        keys = spec.get("keys")
        if keys is not None and not isinstance(keys, dict):
            raise CatalogError(f"{source}: links.{link_name}.keys must be a table mapping left->right")

        through = _opt_str(source, f"links.{link_name}.through", spec.get("through"))
        left_to_through = spec.get("left_to_through")
        through_to_right = spec.get("through_to_right")
        if left_to_through is not None and not isinstance(left_to_through, dict):
            raise CatalogError(f"{source}: links.{link_name}.left_to_through must be a table")
        if through_to_right is not None and not isinstance(through_to_right, dict):
            raise CatalogError(f"{source}: links.{link_name}.through_to_right must be a table")

        validity = None
        validity_raw = spec.get("validity")
        if validity_raw is not None:
            if not isinstance(validity_raw, dict):
                raise CatalogError(f"{source}: links.{link_name}.validity must be a table")
            validity = LinkValiditySpec(
                start_col=_req_str(source, f"links.{link_name}.validity.start_col", validity_raw.get("start_col")),
                end_col=_req_str(source, f"links.{link_name}.validity.end_col", validity_raw.get("end_col")),
                panel_date_col=_req_str(source, f"links.{link_name}.validity.panel_date_col", validity_raw.get("panel_date_col")),
            )

        filters: dict[str, tuple[str, ...]] = {}
        filters_raw = spec.get("filters")
        if filters_raw is not None:
            if not isinstance(filters_raw, dict):
                raise CatalogError(f"{source}: links.{link_name}.filters must be a table")
            for table, preds in filters_raw.items():
                filters[table] = tuple(_req_str_list(source, f"links.{link_name}.filters.{table}", preds))

        prefer: dict[str, tuple[str, ...]] = {}
        prefer_raw = spec.get("prefer")
        if prefer_raw is not None:
            if not isinstance(prefer_raw, dict):
                raise CatalogError(f"{source}: links.{link_name}.prefer must be a table")
            for col, vals in prefer_raw.items():
                prefer[col] = tuple(_req_str_list(source, f"links.{link_name}.prefer.{col}", vals))

        res[link_name] = LinkSpec(
            name=link_name,
            left_table=left_table,
            right_table=right_table,
            keys=keys or {},
            through=through,
            left_to_through=left_to_through,
            through_to_right=through_to_right,
            validity=validity,
            filters=filters,
            prefer=prefer,
        )

    return res


def load_feature_catalog_from_dict(data: Mapping[str, Any], *, source: str) -> dict[str, FeatureSpec]:
    _validate_version(data, source)
    features = data.get("features")
    if not isinstance(features, dict) or len(features) == 0:
        raise CatalogError(f"{source}: missing or empty [features] section")

    res: dict[str, FeatureSpec] = {}
    for feat_name, spec in features.items():
        if not isinstance(spec, dict):
            raise CatalogError(f"{source}: features.{feat_name} must be a table")

        kind = _req_str(source, f"features.{feat_name}.kind", spec.get("kind"))
        if kind not in ("raw", "derived"):
            raise CatalogError(f"{source}: features.{feat_name}.kind must be 'raw' or 'derived' (got {kind!r})")

        if kind == "raw":
            table = _req_str(source, f"features.{feat_name}.table", spec.get("table"))
            cols = _req_str_list(source, f"features.{feat_name}.columns", spec.get("columns"))
            if len(cols) == 0:
                raise CatalogError(f"{source}: features.{feat_name}.columns must be non-empty")
            res[feat_name] = FeatureSpec(name=feat_name, kind="raw", table=table, columns=tuple(cols))
            continue

        builder = _req_str(source, f"features.{feat_name}.builder", spec.get("builder"))
        inputs_raw = spec.get("inputs")
        if not isinstance(inputs_raw, list) or len(inputs_raw) == 0:
            raise CatalogError(f"{source}: features.{feat_name}.inputs must be a non-empty array")

        inputs: list[FeatureInputSpec] = []
        for idx, item in enumerate(inputs_raw):
            if not isinstance(item, dict):
                raise CatalogError(f"{source}: features.{feat_name}.inputs[{idx}] must be a table")
            if "feature" in item:
                inputs.append(FeatureRef(feature=_req_str(source, f"features.{feat_name}.inputs[{idx}].feature", item.get("feature"))))
            elif "table" in item:
                table = _req_str(source, f"features.{feat_name}.inputs[{idx}].table", item.get("table"))
                cols = _req_str_list(source, f"features.{feat_name}.inputs[{idx}].columns", item.get("columns"))
                if len(cols) == 0:
                    raise CatalogError(f"{source}: features.{feat_name}.inputs[{idx}].columns must be non-empty")
                inputs.append(TableColumnsRef(table=table, columns=tuple(cols)))
            elif "link" in item:
                inputs.append(LinkRef(link=_req_str(source, f"features.{feat_name}.inputs[{idx}].link", item.get("link"))))
            else:
                raise CatalogError(
                    f"{source}: features.{feat_name}.inputs[{idx}] must contain one of: 'feature', 'table', or 'link'"
                )

        res[feat_name] = FeatureSpec(name=feat_name, kind="derived", builder=builder, inputs=tuple(inputs))

    return res


def load_table_catalog(path: str | Path) -> dict[str, TableSpec]:
    return load_table_catalog_from_dict(_load_toml_dict(path), source=str(path))


def load_link_graph(path: str | Path) -> dict[str, LinkSpec]:
    return load_link_graph_from_dict(_load_toml_dict(path), source=str(path))


def load_feature_catalog(path: str | Path) -> dict[str, FeatureSpec]:
    return load_feature_catalog_from_dict(_load_toml_dict(path), source=str(path))


def validate_catalog_bundle(
    *,
    tables: Mapping[str, TableSpec],
    links: Mapping[str, LinkSpec],
    features: Mapping[str, FeatureSpec],
) -> None:
    # Feature table references
    for feat in features.values():
        if feat.kind == "raw":
            if feat.table not in tables:
                raise CatalogError(f"Feature '{feat.name}' references unknown table '{feat.table}'.")
        else:
            for inp in feat.inputs:
                if isinstance(inp, TableColumnsRef) and inp.table not in tables:
                    raise CatalogError(f"Feature '{feat.name}' inputs reference unknown table '{inp.table}'.")
                if isinstance(inp, LinkRef) and inp.link not in links:
                    raise CatalogError(f"Feature '{feat.name}' references unknown link '{inp.link}'.")
                if isinstance(inp, FeatureRef) and inp.feature not in features:
                    raise CatalogError(f"Feature '{feat.name}' references unknown feature '{inp.feature}'.")

    # Link table references
    for link in links.values():
        for t in (link.left_table, link.right_table):
            if t not in tables:
                raise CatalogError(f"Link '{link.name}' references unknown table '{t}'.")
        if link.through is not None and link.through not in tables:
            raise CatalogError(f"Link '{link.name}' references unknown through table '{link.through}'.")


def load_bundle(
    *,
    catalog_path: str | Path,
    links_path: str | Path,
    features_path: str | Path,
) -> WrdsCatalogBundle:
    tables = load_table_catalog(catalog_path)
    links = load_link_graph(links_path)
    features = load_feature_catalog(features_path)
    validate_catalog_bundle(tables=tables, links=links, features=features)
    return WrdsCatalogBundle(tables=tables, links=links, features=features)


def _validate_version(data: Mapping[str, Any], path: str | Path) -> None:
    v = data.get("version")
    if v != 1:
        raise CatalogError(f"{path}: expected version = 1 (got {v!r})")


def _req_str(path: str | Path, key: str, value: Any) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise CatalogError(f"{path}: '{key}' must be a non-empty string")
    return value


def _opt_str(path: str | Path, key: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        raise CatalogError(f"{path}: '{key}' must be a string")
    return value


def _req_str_list(path: str | Path, key: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise CatalogError(f"{path}: '{key}' must be an array of strings")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or item.strip() == "":
            raise CatalogError(f"{path}: '{key}[{i}]' must be a non-empty string")
        out.append(item)
    return out


def _opt_str_list(path: str | Path, key: str, value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    return _req_str_list(path, key, value)
