from __future__ import annotations

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


CARE_DERIVED_SQL_FILES = (
    "01_sofa_hourly.sql",
    "02_occult_hypoperfusion_slice.sql",
    "03_sofa_labels_6_12.sql",
)

REQUIRED_UPSTREAM_TABLES = (
    ("mimiciv_derived", "sofa"),
    ("mimiciv_derived", "icustay_hourly"),
    ("mimiciv_icu", "chartevents"),
)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_db_path(repo_root: Path) -> Path:
    db_path = Path(os.getenv("MIMIC_DB_PATH", str(repo_root / "data" / "mimiciv.duckdb"))).expanduser()
    if not db_path.is_absolute():
        db_path = (repo_root / db_path).resolve()
    return db_path


def _table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    count = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchone()[0]
    return bool(count)


def materialize_care_derived_tables(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
) -> None:
    missing = [
        f"{schema}.{table}"
        for schema, table in REQUIRED_UPSTREAM_TABLES
        if not _table_exists(con, schema, table)
    ]
    if missing:
        raise RuntimeError(
            "Missing upstream MIMIC tables required by CARE: " + ", ".join(missing)
        )

    sql_dir = repo_root / "sql" / "care_derived"
    for filename in CARE_DERIVED_SQL_FILES:
        sql_path = sql_dir / filename
        if not sql_path.exists():
            raise FileNotFoundError(f"Missing CARE derived-table SQL: {sql_path}")
        logger.info(f"Materializing CARE derived table with {sql_path.name}")
        con.execute(sql_path.read_text(encoding="utf-8"))

    logger.success("CARE derived tables materialized successfully")


def build_care_derived_tables() -> None:
    repo_root = resolve_repo_root()
    db_path = resolve_db_path(repo_root)
    if not db_path.exists():
        raise FileNotFoundError(
            f"MIMIC-IV DuckDB database not found: {db_path}. Run python/build_mimic_db.py first."
        )

    logger.info(f"Target DB: {db_path}")
    with duckdb.connect(str(db_path)) as con:
        materialize_care_derived_tables(con, repo_root)


if __name__ == "__main__":
    build_care_derived_tables()
