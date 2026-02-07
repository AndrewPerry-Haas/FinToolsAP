"""
FinToolsAP.WebData.registry
============================

Central registry that maps human-readable characteristic names
(e.g. ``'me'``, ``'book_to_market'``, ``'ret'``) to Python callables and
the raw WRDS columns those callables require.

Design goals
------------
* **Uniform treatment** – "cleaning" a raw column and "creating" a derived
  characteristic use the exact same interface.
* **Aliasing** – a function that cleans raw returns can be mapped to the
  name ``'ret'`` (overwriting the raw column) *or* ``'ret_clean'`` (creating
  a new column alongside the raw one).
* **Multi-table source spec** – each function carries a ``.needs`` dict
  mapping ``schema.table`` → ``[columns]`` so the engine can aggregate
  fetches across characteristics.
* **User extensibility** – user-defined characteristics are auto-discovered
  from ``~/.fintoolsap/custom_chars.py`` at import time.

Public API
----------
``REGISTRY`` – the singleton :class:`CharRegistry` populated with built-in
and (optionally) user-defined characteristics.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Characteristic:
    """Immutable descriptor for one output column.

    Parameters
    ----------
    name : str
        The output column name that will appear in the final DataFrame
        (e.g. ``'me'``, ``'ret'``, ``'book_to_market'``).
    func : Callable[[dict[str, pandas.DataFrame], str], pandas.Series]
        A callable with signature ``(raw_tables: dict[str, DataFrame], freq: str) -> Series``.
        ``raw_tables`` is ``{'wrds_table_alias': DataFrame, ...}`` – the
        cleaned, merged panel keyed by ``(permco, date)`` or equivalent.
    dependencies : dict[str, list[str]]
        ``{schema.table: [col1, col2, ...]}`` – the raw columns this
        characteristic needs to be present in the fetched data.
    description : str
        Human-readable docstring for the characteristic.
    order : int
        Explicit execution priority (lower = earlier).  Characteristics with
        the same order are executed in registration order.
    """
    name: str
    func: Callable[..., pandas.Series]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    description: str = ""
    order: int = 100


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CharRegistry:
    """A name → :class:`Characteristic` mapping with helper methods.

    The registry is the **single source of truth** about what characteristics
    the engine can produce.  It is populated in three stages:

    1. Built-in definitions shipped in ``FinToolsAP.WebData.definitions.*``.
    2. (Optional) user definitions from ``~/.fintoolsap/custom_chars.py``.
    3. Programmatic ``register()`` calls at runtime.
    """

    def __init__(self) -> None:
        self._chars: dict[str, Characteristic] = {}

    # -- mutation -----------------------------------------------------------

    def register(self, char: Characteristic) -> None:
        """Add or overwrite a characteristic in the registry.

        If a characteristic with the same ``name`` already exists it is
        silently replaced – this is the aliasing / override mechanism.
        """
        if not isinstance(char, Characteristic):
            raise TypeError(
                f"Expected a Characteristic instance, got {type(char).__name__}"
            )
        self._chars[char.name] = char
        logger.debug("Registered characteristic: %s", char.name)

    def register_func(
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        order: int = 100,
    ) -> None:
        """Convenience wrapper: register a bare function that carries a
        ``.needs`` attribute.

        Parameters
        ----------
        func : callable
            Must have a ``.needs`` attribute (``dict[str, list[str]]``).
        name : str, optional
            Output column name.  Defaults to ``func.__name__``.
        order : int
            Execution priority (lower = earlier).
        """
        needs: dict[str, list[str]] = getattr(func, "needs", None)
        if needs is None:
            raise ValueError(
                f"Function {func.__name__!r} is missing a '.needs' attribute."
            )
        out_name = name or func.__name__
        char = Characteristic(
            name=out_name,
            func=func,
            dependencies=needs,
            description=(func.__doc__ or "").strip(),
            order=order,
        )
        self.register(char)

    def unregister(self, name: str) -> None:
        """Remove a characteristic by name (no-op if missing)."""
        self._chars.pop(name, None)

    # -- query --------------------------------------------------------------

    def get(self, name: str) -> Characteristic:
        """Return the :class:`Characteristic` for *name*, or raise ``KeyError``."""
        try:
            return self._chars[name]
        except KeyError:
            raise KeyError(
                f"Unknown characteristic {name!r}. "
                f"Available: {sorted(self._chars)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._chars

    def __len__(self) -> int:
        return len(self._chars)

    def available(self) -> list[str]:
        """Return a sorted list of all registered characteristic names."""
        return sorted(self._chars)

    def resolve(self, names: list[str]) -> list[Characteristic]:
        """Look up multiple names and return them sorted by execution order.

        Raises ``KeyError`` for any unknown name.
        """
        chars = [self.get(n) for n in names]
        chars.sort(key=lambda c: c.order)
        return chars

    def aggregate_needs(self, chars: list[Characteristic]) -> dict[str, set[str]]:
        """Merge the ``.dependencies`` dicts of several characteristics.

        Returns ``{table: {col1, col2, ...}}`` – the union of all required
        raw columns grouped by table.
        """
        merged: dict[str, set[str]] = {}
        for c in chars:
            for table, cols in c.dependencies.items():
                merged.setdefault(table, set()).update(cols)
        return merged

    def describe(self, name: Optional[str] = None) -> dict[str, str]:
        """Return ``{name: description}`` for one or all characteristics."""
        if name is not None:
            c = self.get(name)
            return {c.name: c.description}
        return {n: c.description for n, c in sorted(self._chars.items())}


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _collect_chars_from_module(module: types.ModuleType) -> list[Callable]:
    """Return all callables in *module* that carry a ``.needs`` attribute."""
    found: list[Callable] = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if callable(obj) and hasattr(obj, "needs") and not attr_name.startswith("_"):
            found.append(obj)
    return found


def load_builtins(registry: CharRegistry) -> None:
    """Import every module in ``WebData.definitions`` (or
    ``FinToolsAP.WebData.definitions``) and register the annotated functions
    found there."""
    from . import definitions  # noqa: E402
    pkg_path = Path(definitions.__file__).parent
    # Derive the real package prefix at runtime so it works whether the
    # top-level package is ``FinToolsAP`` or just ``WebData``.
    defs_pkg = definitions.__name__  # e.g. "WebData.definitions" or "FinToolsAP.WebData.definitions"

    for py_file in sorted(pkg_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        mod_name = f"{defs_pkg}.{py_file.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            logger.warning("Failed to import built-in definitions from %s", mod_name, exc_info=True)
            continue
        for func in _collect_chars_from_module(mod):
            # Functions may declare an _output_name to alias themselves
            out_name = getattr(func, "_output_name", None) or func.__name__
            order = getattr(func, "_order", 100)
            registry.register_func(func, name=out_name, order=order)


def load_user_plugins(registry: CharRegistry) -> None:
    """Auto-discover ``~/.fintoolsap/custom_chars.py`` and register any
    valid characteristic functions found inside it."""
    user_file = Path.home() / ".fintoolsap" / "custom_chars.py"
    if not user_file.is_file():
        logger.debug("No user plugin file at %s", user_file)
        return

    logger.info("Loading user characteristics from %s", user_file)
    spec = importlib.util.spec_from_file_location("_fintoolsap_user_chars", str(user_file))
    if spec is None or spec.loader is None:
        logger.warning("Could not build import spec for %s", user_file)
        return
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        logger.warning("Error executing user plugin %s", user_file, exc_info=True)
        return

    for func in _collect_chars_from_module(mod):
        out_name = getattr(func, "_output_name", None) or func.__name__
        order = getattr(func, "_order", 100)
        registry.register_func(func, name=out_name, order=order)
        logger.info("User characteristic registered: %s", out_name)


# ---------------------------------------------------------------------------
# Decorator for defining characteristics
# ---------------------------------------------------------------------------

def characteristic(
    *,
    name: Optional[str] = None,
    needs: Optional[Dict[str, List[str]]] = None,
    order: int = 100,
) -> Callable:
    """Decorator to annotate a function as a characteristic definition.

    Usage
    -----
    ::

        @characteristic(name='me', needs={'crsp.msf': ['prc', 'shrout']}, order=20)
        def calc_market_equity(raw_tables, freq):
            sf = raw_tables['crsp.msf']
            return sf['prc'].abs() * sf['shrout']

    The decorated function retains its ``.needs``, ``._output_name``, and
    ``._order`` attributes so the registry loader can pick them up.
    """
    def decorator(func: Callable) -> Callable:
        func.needs = needs or {}
        func._output_name = name or func.__name__
        func._order = order
        return func
    return decorator


# ---------------------------------------------------------------------------
# Singleton registry – populated on first import
# ---------------------------------------------------------------------------

REGISTRY = CharRegistry()


def _bootstrap() -> None:
    """Populate the global REGISTRY with built-in + user definitions.

    Called once at module import time.
    """
    load_builtins(REGISTRY)
    load_user_plugins(REGISTRY)


_bootstrap()
