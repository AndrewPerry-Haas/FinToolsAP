#!/usr/bin/env bash
# import_csv.sh  —  fast CSV → SQLite loader
# Usage: ./import_csv.sh /path/to/db.sqlite /path/to/data.csv target_table

set -euo pipefail

DB_PATH="${1:?missing database path}"
CSV_PATH="${2:?missing CSV path}"
TABLE="${3:?missing table name}"

sqlite3 "$DB_PATH" <<SQL
-- Silence the “off” echoed by the PRAGMAs
.output /dev/null
PRAGMA journal_mode = OFF;
PRAGMA synchronous  = OFF;
.output stdout

-- Import the CSV (expects header row)
.mode csv
.import '$CSV_PATH' $TABLE
.mode columns
.quit
SQL
