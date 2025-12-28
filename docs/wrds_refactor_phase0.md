# WRDS Refactor Phase 0: Reconnaissance

## Current entrypoints

- Public WRDS feature API: `FinToolsAP.WebData.WebData.getData()` in `src/FinToolsAP/WebData.py`.
- Separate “bulk download into SQLite” path lives in `src/FinToolsAP/LocalDatabase.py` and related helpers; it is *not* the characteristic/builder pipeline.

## Current architecture inside `WebData`

### Connection/auth

- `WebData.__init__(username)` creates a `wrds.Connection(username=...)`.
- Credential prompting and `.pgpass` creation are handled by `wrds` (non-goal: unchanged).

### Dependency map + builders

- `WebData._dep_map`: maps output fields (e.g., `dp`, `bm`) to:
  - required logical sources: `tables` (e.g., `SF`, `SE`, `COMP`, `LINK`, `INDEX`)
  - required columns per source group (`crsp_sf_cols`, `crsp_se_cols`, `comp_cols`, `index_cols`)
  - required builder names (`builders`).

- `WebData._builder_registry`: maps builder names to bound methods.
- `WebData._builder_order`: dependency-safe ordering to apply builders.

### Fetch layer

- Unified loader: `WebData._load_wrds(table_alias, columns, start_date, end_date, ...)`.
  - Uses `_build_sql_string()` for projection: `SELECT <cols> FROM <schema.table> WHERE <date> BETWEEN ...`.
  - Supports optional `predicates` and `id_type IN (...)` constraints.

- Alias mapping in `_load_wrds()` includes:
  - `CRSP.SEALL.M|D` → `CRSP.MSEALL` / `CRSP.DSEALL`
  - `CRSP.SF.M|D` → `CRSP.MSF` / `CRSP.DSF`
  - `CRSP.SI.M|D` → `CRSP.MSI` / `CRSP.DSI`
  - `CRSP.LINK` → `CRSP.CCMXPF_LINKTABLE`
  - `COMP.FUNDQ` → `COMP.FUNDQ`

### Merge / join flow (intended)

The intended orchestration (currently present but unreachable) is:

1. Load CRSP “SE” identity (ticker→permco, etc.) via `_load_se_data()`.
2. Extract `permco` universe; load CRSP “SF” time-series via `_load_sf_data()`.
3. Merge SF+SE on `(date, permco)`.
4. If Compustat features are required:
   - Load CCM link table via `_load_ccm_link_data()`.
   - Join on `permco` and apply date-validity bounds (`linkdt`/`linkenddt`).
   - Load Compustat FUNDQ by `gvkey` via `_load_comp_data()`.
   - Merge Compustat onto panel and forward-fill.
5. If index series required: load and merge CRSP index by `date`.

### Central cleaning

- `_clean_inputs()` normalizes date types and does periodic alignment for SE/SF/COMP/INDEX.

## Builder inventory (current internal methods)

All builder methods currently live in `src/FinToolsAP/WebData.py` and expect a “panel” DataFrame keyed by `permco` and `date`.

- `split_adjust`: uses `cfacpr`, `cfacshr` to split-adjust `prc`, `bidlo`, `askhi` and scale `shrout`.
- `build_me`: `me = prc * shrout`.
- `build_div`: `div = (ret - retx) * prc.shift(1)`.
- `build_dp`: rolling sum of `div` by `permco`, then `dp = div_12m_sum / prc`.
- `build_dps`: `dps = div_12m_sum / shrout`.
- `build_pps`: `pps = prc / shrout`.
- `build_be`, `build_earn`: currently stubs (no calculation yet).
- `build_earn_ann`: annualizes `earn` with a rolling 4-quarter sum.
- `build_bm`, `build_bps`, `build_ep`, `build_eps`: ratio builders.

## Current tests

- `tests/TEST_WebData_unit_test.py` defines `MockWebData(WebData)` overriding loader methods to return synthetic frames.
  - Tests include:
    - dp min-periods behavior (monthly/daily)
    - link overlap does not break monthly coverage
    - “do not add unrequested ratio columns”

- `tests/TEST_WebData.py` appears integration-style (real WRDS).

## High-impact pitfalls to fix during refactor

- `WebData.getData()` currently prints debug state then executes a bare `raise`, making the pipeline unusable.
- Duplicate logic exists between `_clean_crsp_data()` (legacy monolith) and the builder methods.
- `build_div()` uses `prc.shift(1)` without grouping by `permco` (risk of cross-security bleed if the panel has multiple permcos).
- Some builders create intermediate columns (e.g., `div_12m_sum`) that should not be returned unless requested.
