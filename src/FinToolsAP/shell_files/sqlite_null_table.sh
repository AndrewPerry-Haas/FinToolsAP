#!/usr/bin/env bash
set -euo pipefail

db="$1"       
table="$2"   

build_update_sql() {
  # Collect column names with PRAGMA table_info
  mapfile -t cols < <(
    sqlite3 "$db" -noheader -csv \
      "PRAGMA table_info(\"$table\");" | awk -F',' '{print $2}'
  )
  [[ ${#cols[@]} -eq 0 ]] && { echo "-- table empty? nothing to do"; return; }

  # Construct SET and WHERE clauses
  local set_clause="" where_clause=""
  for c in "${cols[@]}"; do
    set_clause+="\"$c\" = NULLIF(TRIM(\"$c\"), ''), "
    where_clause+="\"$c\" = '' OR "
  done
  set_clause="${set_clause%, }"      # drop trailing comma+space
  where_clause="${where_clause% OR }"

  echo "UPDATE \"$table\" SET $set_clause WHERE $where_clause;"
}

{
  echo "BEGIN;"
  build_update_sql
  echo "COMMIT;"
} | sqlite3 "$db"

echo "✓ Empty strings in $table replaced with NULL."
