import glob
import os
import re
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from loguru import logger

from build_care_derived_tables import materialize_care_derived_tables

load_dotenv()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_raw_csv_dir(repo_root: Path) -> Path:
    raw_dir_env = os.getenv("MIMIC_RAW_CSV_DIR", "").strip()
    if not raw_dir_env:
        raise RuntimeError(
            "MIMIC_RAW_CSV_DIR is not set. Point it to your licensed MIMIC-IV CSV root, "
            "for example: /path/to/mimiciv/3.1"
        )
    raw_dir = Path(raw_dir_env)
    if not raw_dir.is_absolute():
        raw_dir = (repo_root / raw_dir).resolve()
    return raw_dir


def resolve_mimic_code_root(repo_root: Path) -> Path:
    code_dir_env = os.getenv("MIMIC_CODE_DIR", "").strip()
    if not code_dir_env:
        raise RuntimeError(
            "MIMIC_CODE_DIR is not set. Point it to your external mimic-code checkout, "
            "for example: /path/to/mimic-code"
        )
    code_dir = Path(code_dir_env)
    if not code_dir.is_absolute():
        code_dir = (repo_root / code_dir).resolve()
    return code_dir


def apply_duckdb_schema_fixes(sql_text: str) -> str:
    sql_text = re.sub(r"TIMESTAMP\([0-9]+\)", "TIMESTAMP", sql_text)
    sql_text = re.sub(r"spec_type_desc(.+)NOT NULL", r"spec_type_desc\1", sql_text)
    sql_text = re.sub(r"drug +(VARCHAR.+)NOT NULL", r"drug \1", sql_text)
    return sql_text


def build_database() -> None:
    repo_root = resolve_repo_root()
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(os.getenv("MIMIC_DB_PATH", str(data_dir / "mimiciv.duckdb"))).expanduser()
    if not db_path.is_absolute():
        db_path = (repo_root / db_path).resolve()

    raw_csv_dir = resolve_raw_csv_dir(repo_root)
    mimic_code_root = resolve_mimic_code_root(repo_root)
    concepts_dir = mimic_code_root / "mimic-iv" / "concepts_duckdb"
    create_sql_path = mimic_code_root / "mimic-iv" / "buildmimic" / "postgres" / "create.sql"

    if not create_sql_path.exists():
        raise FileNotFoundError(f"Missing create.sql: {create_sql_path}")
    if not (concepts_dir / "duckdb.sql").exists():
        raise FileNotFoundError(f"Missing duckdb.sql: {concepts_dir / 'duckdb.sql'}")
    if not raw_csv_dir.exists():
        raise FileNotFoundError(f"Raw MIMIC-IV CSV root not found: {raw_csv_dir}")

    logger.info(f"Target DB: {db_path}")
    logger.info(f"Raw MIMIC-IV CSV root: {raw_csv_dir}")
    logger.info(f"mimic-code root: {mimic_code_root}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = data_dir / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA memory_limit='16GB';")
        con.execute(f"PRAGMA temp_directory='{temp_dir}';")
        con.execute("PRAGMA threads=4;")

        logger.info("Step 0: applying canonical MIMIC-IV schema from create.sql")
        create_sql = create_sql_path.read_text(encoding="utf-8")
        con.execute(apply_duckdb_schema_fixes(create_sql))

        logger.info("Step 1: ingesting base CSV.GZ tables into mimiciv_hosp and mimiciv_icu")
        csv_files = sorted(glob.glob(str(raw_csv_dir / "*" / "*.csv.gz")))
        if not csv_files:
            raise RuntimeError(f"No .csv.gz files found under {raw_csv_dir}/*/")

        loaded = 0
        for filepath in csv_files:
            schema_dir = Path(filepath).parent.name
            if schema_dir not in {"hosp", "icu"}:
                continue
            table_name = Path(filepath).name.split(".")[0]
            full_table_name = f"mimiciv_{schema_dir}.{table_name}"
            logger.info(f"Loading {full_table_name}")
            try:
                con.execute(
                    f"COPY {full_table_name} FROM '{filepath}' (HEADER, DELIM ',', QUOTE '\"', ESCAPE '\"');"
                )
                loaded += 1
            except Exception as exc:
                if "does not exist" in str(exc):
                    logger.warning(f"Skipped {full_table_name}: table missing from schema definition")
                    continue
                raise

        logger.info(f"Loaded {loaded} base tables")

        logger.info("Step 2: compiling official mimiciv_derived concepts from concepts_duckdb/duckdb.sql")
        con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived;")
        old_cwd = Path.cwd()
        os.chdir(concepts_dir)
        try:
            lines = (concepts_dir / "duckdb.sql").read_text(encoding="utf-8").splitlines()
            sql_commands = [line.strip() for line in lines if line.strip().startswith(".read ")]
            for line in sql_commands:
                sql_file = line.replace(".read ", "", 1).strip()
                logger.info(f"Compiling {sql_file}")
                con.execute(Path(sql_file).read_text(encoding="utf-8"))
        finally:
            os.chdir(old_cwd)

        logger.info("Step 3: materializing CARE-specific derived tables")
        materialize_care_derived_tables(con, repo_root)

        logger.success("DuckDB build completed successfully")
        logger.success("Schemas available: mimiciv_hosp, mimiciv_icu, mimiciv_derived")
        logger.success(f"Database path: {db_path}")
    finally:
        con.close()


if __name__ == "__main__":
    build_database()
