# WRDS User Guide (TOML + Plugins)

This guide explains how `FinToolsAP.WebData.WebData.getData()` decides what to fetch from WRDS, how to permanently customize feature definitions using TOML files, and how to add new derived features via Python plugins.

## Quick start

Minimal example (monthly data):

```python
from FinToolsAP.WebData import WebData

wd = WebData("<your_wrds_username>")

df = wd.getData(
    tickers=["AAPL"],
    fields=["bm", "me", "dp"],
    start_date="2015-01-01",
    end_date="2020-12-31",
    freq="M",
)

print(df.head())
```

Notes:
- `getData()` always includes `ticker`, `date`, and `permco` in the returned DataFrame (even if you don’t request them).
- Setting `tickers=None` triggers a universe pull and can be extremely large.

## What changed (high level)

WebData’s WRDS pipeline is now:
- TOML-driven: data sources, link rules, and feature definitions live in persistent TOML files.
- Planned at runtime: before fetching data, an execution plan is built that:
  - projects only the needed columns from each WRDS table
  - performs joins only if some requested feature needs them
  - computes derived features in dependency order
- Extensible: derived feature builders can be installed as plugins via Python entry points.

## Where the TOML files live

FinToolsAP ships a default, versioned TOML bundle inside the package:
- `src/FinToolsAP/wrds/data/wrds_catalog.toml`
- `src/FinToolsAP/wrds/data/wrds_links.toml`
- `src/FinToolsAP/wrds/data/wrds_features.toml`

When FinToolsAP is installed from a wheel, these files are packaged inside the distribution and loaded via `importlib.resources`.

## How to override TOMLs permanently

There are two supported ways to make changes “stick” across runs:

### Option A (recommended): copy the default TOMLs and set env vars

This is the typical workflow when you installed FinToolsAP via `pip` and want local, persistent customization.

1) Install FinToolsAP into your environment.

```bash
pip install FinToolsAP
```

1) Copy the packaged TOMLs into a directory you own (example path: `~/.config/FinToolsAP/wrds/`).

2) Set these environment variables to point to your edited TOMLs:
- `WRDS_CATALOG_PATH`
- `WRDS_LINKS_PATH`
- `WRDS_FEATURES_PATH`

Example (bash):

```bash
export WRDS_CATALOG_PATH="$HOME/.config/FinToolsAP/wrds/wrds_catalog.toml"
export WRDS_LINKS_PATH="$HOME/.config/FinToolsAP/wrds/wrds_links.toml"
export WRDS_FEATURES_PATH="$HOME/.config/FinToolsAP/wrds/wrds_features.toml"
```

From then on, `WebData.getData()` will use your TOMLs automatically.

Getting the default TOMLs onto disk (one-time helper snippet):

```python
from pathlib import Path
import importlib.resources as r

out_dir = Path.home() / ".config" / "FinToolsAP" / "wrds"
out_dir.mkdir(parents=True, exist_ok=True)

pkg = "FinToolsAP.wrds.data"

for name in ("wrds_catalog.toml", "wrds_links.toml", "wrds_features.toml"):
    src = r.files(pkg).joinpath(name)
    dst = out_dir / name
    dst.write_bytes(src.read_bytes())
    print("wrote", dst)
```

3) Export the environment variables (and optionally add them to `~/.bashrc` or your shell profile so they persist across terminals).

```bash
export WRDS_CATALOG_PATH="$HOME/.config/FinToolsAP/wrds/wrds_catalog.toml"
export WRDS_LINKS_PATH="$HOME/.config/FinToolsAP/wrds/wrds_links.toml"
export WRDS_FEATURES_PATH="$HOME/.config/FinToolsAP/wrds/wrds_features.toml"
```

### Option B (advanced): instantiate `WrdsEngine` with explicit paths

If you want to bypass environment variables and point at specific files programmatically:

```python
from FinToolsAP.WebData import WebData
from FinToolsAP.wrds.engine import WrdsEngine, EngineConfig

wd = WebData("<your_wrds_username>")

engine = WrdsEngine(
    fetch_se=wd._load_se_data,
    fetch_sf=wd._load_sf_data,
    fetch_index=wd._load_index_data,
    fetch_link=wd._load_ccm_link_data,
    fetch_comp=wd._load_comp_data,
    clean_inputs=wd._clean_inputs,
    config=EngineConfig(
        catalog_path="/path/to/wrds_catalog.toml",
        links_path="/path/to/wrds_links.toml",
        features_path="/path/to/wrds_features.toml",
    ),
)

df = engine.get(
    features=["ticker", "date", "permco", "bm"],
    tickers=["AAPL"],
    freq="M",
    start_date="2015-01-01",
    end_date="2020-12-31",
    include_identity=False,
)
```

## Understanding `wrds_features.toml`

`wrds_features.toml` is the primary file you will edit.

A feature is either:
- `kind = "raw"`: a column fetched directly from a table
- `kind = "derived"`: computed by a builder function, with explicit dependencies

### Raw feature example

```toml
[features.ret]
kind = "raw"
table = "crsp_sf"
columns = ["ret"]
```

### Derived feature example

```toml
[features.me]
kind = "derived"
builder = "build_me"
inputs = [
  { feature = "prc" },
  { feature = "shrout" },
]
```

### Forcing a join (link dependency)

Some features require joining CRSP to Compustat via the CCM link table. You express this by including a `link` input:

```toml
[features.be]
kind = "derived"
builder = "build_be"
inputs = [
  { link = "crsp_comp" },
  { table = "comp_fundq", columns = ["gvkey", "seqq", "txditcq", "pstkrq", "pstkq"] },
]
```

The planner will only include the `crsp_comp` join if at least one requested feature depends on it.

## Understanding `wrds_catalog.toml` and `wrds_links.toml`

- `wrds_catalog.toml` names the canonical internal tables (e.g., `crsp_sf`, `comp_fundq`) and maps them to WRDS schema/table names.
- `wrds_links.toml` defines possible join edges (e.g., `crsp_comp` through `ccm_link`) and their validity rules.

Most users should not need to edit these unless they are changing schemas/tables, or modifying link logic.

## Adding a new derived feature (no Python)

If you can build your feature using existing builders, you only need to edit TOML.

Example: expose a new alias feature `log_me` implemented by a plugin builder named `log1p` (see next section). In `wrds_features.toml`:

```toml
[features.log_me]
kind = "derived"
builder = "log1p"
inputs = [
  { feature = "me" },
]
```

## Adding a new derived feature (Python plugin)

FinToolsAP discovers plugin builders via Python entry points:
- **Entry point group**: `fintoolsap_wrds.builders`
- **Entry point name**: the builder name you reference from TOML
- **Entry point object**: a callable with signature:

```python
(inputs: dict[str, object], ctx) -> object
```

Where:
- `inputs` contains the requested inputs (Series/DataFrames)
- `ctx.feature` is the feature name currently being built
- `ctx.base_index` is the expected `(permco, date)` MultiIndex for Series outputs

### Minimal plugin package example

Create a small package (e.g., `fintoolsap_wrds_extra/`).

`pyproject.toml`:

```toml
[project]
name = "fintoolsap-wrds-extra"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["FinToolsAP"]

[project.entry-points."fintoolsap_wrds.builders"]
log1p = "fintoolsap_wrds_extra.builders:log1p"
```

`fintoolsap_wrds_extra/builders.py`:

```python
import numpy as np
import pandas as pd


def log1p(inputs, ctx):
    # Single-input builder: accepts feature name or a conventional key.
    # Here we just take the first input value.
    x = next(iter(inputs.values()))
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("log1p expects a 1-col DataFrame or Series")
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        raise ValueError("log1p expects a Series")

    y = np.log1p(x.astype(float))
    # return Series aligned to the engine’s base index
    return y.reindex(ctx.base_index)
```

Install it (editable install during development):

```bash
pip install -e /path/to/fintoolsap_wrds_extra
```

Then define a feature in your `wrds_features.toml` referencing `builder = "log1p"`.

## Worked example: add a new characteristic `cfp`

Goal: create a new characteristic called `cfp` that:
- pulls `saleq` from Compustat quarterly fundamentals (`comp_fundq`)
- links Compustat → CRSP (so we can align it to the CRSP panel)
- uses CRSP `me` (market equity)
- computes $\text{cfp} = \frac{\sum_{i=0}^{3} \text{saleq}_{t-i}}{\text{me}_t}$

Assumption: “rolling 4 quarter sum” means the last 4 *Compustat quarters* for each `gvkey`, not a rolling window over monthly/daily observations.

### Step 1: edit your local `wrds_features.toml`

Open the TOML you copied (the one pointed to by `WRDS_FEATURES_PATH`) and add:

```toml
[features.cfp]
kind = "derived"
builder = "build_cfp"
inputs = [
  # ensures the planner includes the CCM link + Compustat join
  { link = "crsp_comp" },

  # CRSP market equity (already defined in the default TOML)
  { feature = "me" },

  # inputs from Compustat; engine will also ensure gvkey/date are present
  { table = "comp_fundq", columns = ["gvkey", "saleq"] },
]
```

### Step 2: create a small plugin that provides `build_cfp`

Create a small package *inside your research project* (so it’s version-controlled), then install it into the **same Python environment** that has `FinToolsAP`.

Example layout (your stated setup):

- Conda env that already has FinToolsAP: `/home/andrewperry/miniconda3/envs/MyResearchProject`
- Research project folder: `/home/andrewperry/Documents/MyResearchProject`

Create the plugin project here:

```bash
cd /home/andrewperry/Documents/MyResearchProject

mkdir -p fintoolsap_wrds_cfp/fintoolsap_wrds_cfp
touch fintoolsap_wrds_cfp/fintoolsap_wrds_cfp/__init__.py
```

You should end up with:

```
/home/andrewperry/Documents/MyResearchProject/fintoolsap_wrds_cfp/
    pyproject.toml
    fintoolsap_wrds_cfp/
        __init__.py
        builders.py
```

`pyproject.toml`:

```toml
[project]
name = "fintoolsap-wrds-cfp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["FinToolsAP", "pandas", "numpy"]

[project.entry-points."fintoolsap_wrds.builders"]
build_cfp = "fintoolsap_wrds_cfp.builders:build_cfp"
```

`fintoolsap_wrds_cfp/builders.py`:

```python
import numpy as np
import pandas as pd


def build_cfp(inputs, ctx):
    """Compute CFP = trailing 4-quarter sum(saleq) / current me.

    Expected inputs (per the TOML example):
    - inputs["gvkey"]: Series aligned to ctx.base_index
    - inputs["saleq"]: Series aligned to ctx.base_index (quarterly values forward-filled to the panel)
    - inputs["me"]: Series aligned to ctx.base_index
    """

    gvkey = inputs["gvkey"]
    saleq = inputs["saleq"]
    me = inputs["me"]

    # Build a working frame with explicit permco/date columns.
    panel = pd.DataFrame(
        {
            "gvkey": gvkey.values,
            "saleq": pd.to_numeric(saleq, errors="coerce").values,
            "me": pd.to_numeric(me, errors="coerce").values,
        },
        index=ctx.base_index,
    ).reset_index()

    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["qend"] = panel["date"] + pd.tseries.offsets.QuarterEnd(0)

    # Convert the forward-filled panel series into a true quarterly series per gvkey by taking
    # the last observation inside each quarter.
    q = (
        panel.sort_values(["gvkey", "date"], kind="mergesort")
        .groupby(["gvkey", "qend"], as_index=False)
        .tail(1)
        [["gvkey", "qend", "saleq"]]
    )
    q = q.sort_values(["gvkey", "qend"], kind="mergesort")
    q["saleq4"] = (
        q.groupby("gvkey", group_keys=False)["saleq"]
        .rolling(window=4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Map trailing-4Q values back to the panel quarter bucket, and forward-fill within gvkey
    # so months/days after quarter-end carry the latest available trailing sum.
    panel = panel.merge(q[["gvkey", "qend", "saleq4"]], how="left", on=["gvkey", "qend"])
    panel = panel.sort_values(["gvkey", "date"], kind="mergesort")
    panel["saleq4"] = panel.groupby("gvkey", group_keys=False)["saleq4"].ffill()

    # Compute ratio; protect against divide-by-zero.
    panel["cfp"] = panel["saleq4"] / panel["me"].replace({0.0: np.nan})

    out = panel.set_index(["permco", "date"])["cfp"].sort_index()
    return out.reindex(ctx.base_index)
```

Install it into your conda env (editable install during development):

```bash
conda activate /home/andrewperry/miniconda3/envs/MyResearchProject
pip install -e /home/andrewperry/Documents/MyResearchProject/fintoolsap_wrds_cfp
```

Sanity-check that the entry point is visible:

```bash
python -c "from importlib.metadata import entry_points; print([ep.name for ep in entry_points().select(group='fintoolsap_wrds.builders')])"
```

You should see `build_cfp` in the printed list.

### Step 3: use `cfp` from `WebData.getData()`

```python
from FinToolsAP.WebData import WebData

wd = WebData("<your_wrds_username>")
df = wd.getData(
    tickers=["AAPL"],
    fields=["date", "ticker", "permco", "cfp"],
    start_date="2015-01-01",
    end_date="2020-12-31",
    freq="M",
)

print(df[["date", "ticker", "cfp"]].head())
```

## Inspecting the execution plan (advanced)

If you want to see what will be fetched and computed before actually downloading:

```python
from FinToolsAP.WebData import WebData
from FinToolsAP.wrds.engine import WrdsEngine

wd = WebData("<your_wrds_username>")
engine = WrdsEngine(
    fetch_se=wd._load_se_data,
    fetch_sf=wd._load_sf_data,
    fetch_index=wd._load_index_data,
    fetch_link=wd._load_ccm_link_data,
    fetch_comp=wd._load_comp_data,
    clean_inputs=wd._clean_inputs,
)

plan = engine.explain_plan(["ret", "bm", "dp"], include_identity=True)
print(plan.as_dict())
```

## Troubleshooting

### `AttributeError: module 'wrds' has no attribute 'Connection'`

This almost always means the external `wrds` package is being shadowed by a local module path.

Common cause: adding `src/FinToolsAP` to `PYTHONPATH` (or `sys.path`) instead of `src`.

Correct for this repo:

```bash
export PYTHONPATH=/home/andrewperry/Nextcloud/FinToolsAP/src
```

Avoid:

```bash
# don’t do this
export PYTHONPATH=/home/andrewperry/Nextcloud/FinToolsAP/src/FinToolsAP
```

### Queries are too large / slow

- Prefer specifying `fields` instead of requesting everything.
- Prefer a smaller date range.
- Avoid `tickers=None` unless you really want the full universe.

## Reference

- Plugin loader: `FinToolsAP.wrds.plugin.load_entrypoint_builders` (entry point group `fintoolsap_wrds.builders`).
- Built-in builders registry: `FinToolsAP.wrds.builders.REGISTRY`.
