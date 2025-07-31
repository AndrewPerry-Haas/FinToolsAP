#!/usr/bin/env bash
#
# infer_sqlite_types.sh  <database>  <table> [force_text_col1 force_text_col2 ...]
#
# Prints:  col1:INTEGER,col2:REAL,col3:TEXT
#
set -euo pipefail

db="${1:-}"; shift || true
table="${1:-}"; shift || true
force_text=( "$@" )           # optional list of columns to force to TEXT

if [[ -z $db || -z $table ]]; then
  echo "Usage: $0 <database> <table> [force_text columns...]" >&2
  exit 1
fi

# helper: true if $1 is in ${force_text[*]}
is_forced() {
  local needle=$1
  for x in "${force_text[@]}"; do [[ $x == "$needle" ]] && return 0; done
  return 1
}

# 1 ── list columns ──────────────────────────────────────────────────────
mapfile -t cols < <(
  sqlite3 "$db" -csv "PRAGMA table_info('$table');" | awk -F, '{print $2}'
)

pairs=()
for col in "${cols[@]}"; do
  if is_forced "$col"; then
    pairs+=( "${col}:TEXT" )
    continue
  fi

  inferred_type=$(sqlite3 -batch -noheader "$db" <<SQL
WITH
  trimmed AS ( SELECT TRIM("$col") AS v FROM "$table" ),
  /* any row with a dash after position 1 */
  has_mid_dash AS (
      SELECT 1 FROM trimmed
       WHERE INSTR(v, '-') > 1    -- dash after position 1
       LIMIT 1
  ),
  bad AS (
  /* any non‑blank row that is not digits or digits-dot-digits */
      SELECT 1 FROM trimmed
       WHERE v <> ''
         AND v GLOB '*[A-Za-z]*'
       LIMIT 1
  ),
  leading_zero AS (
      /* any numeric‑looking string that starts with 0 but is not just "0" */
      SELECT 1 FROM trimmed
       WHERE v GLOB '0[0-9]*'
         AND v <> '0'
       LIMIT 1
  ),
  frac AS (
      /* any numeric with a decimal point */
      SELECT 1 FROM trimmed
       WHERE v GLOB '*.*'
         AND (v GLOB '-[0-9]*.[0-9]*'
               OR v GLOB '[0-9]*.[0-9]*')
       LIMIT 1
  ),
  all_numeric AS (
      /* every non‑blank row is digits or digits-dot-digits               */
      SELECT COUNT(*) AS bad_cnt FROM (
        SELECT 1 FROM trimmed
         WHERE v <> ''
           AND v NOT GLOB '-[0-9]*'
           AND v NOT GLOB '-[0-9]*.[0-9]*'
           AND v NOT GLOB '[0-9]*'
           AND v NOT GLOB '[0-9]*.[0-9]*'
      )
  )
SELECT
  CASE
      WHEN EXISTS (SELECT * FROM has_mid_dash) THEN 'TEXT'
      WHEN EXISTS (SELECT * FROM bad)          THEN 'TEXT'
      WHEN EXISTS (SELECT * FROM leading_zero) THEN 'TEXT'
      WHEN (SELECT bad_cnt FROM all_numeric) > 0 THEN 'TEXT'
      WHEN EXISTS (SELECT * FROM frac)         THEN 'REAL'
      ELSE                                         'INTEGER'
  END;
SQL
)

  pairs+=( "${col}:${inferred_type}" )
done

IFS=','; echo "${pairs[*]}"
