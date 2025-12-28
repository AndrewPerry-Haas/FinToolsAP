from __future__ import annotations

from typing import Callable, Iterable, Optional

from importlib import metadata

from .builders import BuilderRegistry, BuilderError


class PluginError(RuntimeError):
    pass


def load_entrypoint_builders(
    *,
    group: str = "fintoolsap_wrds.builders",
    registry: BuilderRegistry,
    allow_override: bool = False,
) -> list[str]:
    """Load and register builder callables from Python entry points.

    Plugin contract:
    - Entry point group: `fintoolsap_wrds.builders`
    - Entry point name: builder name referenced by TOML
    - Entry point object: a callable with signature `(inputs: dict[str, object], ctx) -> object`

    Returns the list of successfully registered builder names.
    """

    eps = metadata.entry_points()

    # Python 3.10+: entry_points() may return an object with .select
    if hasattr(eps, "select"):
        selected = list(eps.select(group=group))  # type: ignore[attr-defined]
    else:  # pragma: no cover
        selected = list(eps.get(group, []))  # type: ignore[assignment]

    loaded: list[str] = []
    for ep in selected:
        try:
            obj = ep.load()
        except Exception as e:
            raise PluginError(f"Failed to load builder entry point '{ep.name}' from {ep.value}: {e}") from e

        if not callable(obj):
            raise PluginError(f"Entry point '{ep.name}' is not callable")

        try:
            registry.register(ep.name, obj, override=allow_override)
        except BuilderError:
            if allow_override:
                raise
            # collision without override: skip
            continue

        loaded.append(ep.name)

    return loaded
