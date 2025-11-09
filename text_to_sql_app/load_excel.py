"""Load a large Excel workbook into a DuckDB temporary table for querying."""

import argparse
from pathlib import Path
import tempfile

import duckdb
import pandas as pd


STORAGE_ROOT = Path.home() / ".antiks"
DATA_PATH = STORAGE_ROOT / "data"
DUCKDB_PATH = DATA_PATH / "sandbox.duckdb"
VECTOR_STORE_PATH = STORAGE_ROOT / "vector_store"


def load_excel_to_duckdb(excel_path: Path, table_name: str, sheet: int = 1) -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DUCKDB_PATH))
    temp_path = None
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet - 1, dtype=str)
        df.columns = [str(col).strip() for col in df.columns]
        df = df.fillna("")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            df.to_parquet(tmp_file.name, index=False)
            temp_path = tmp_file.name

        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{temp_path}');")
    except Exception as exc:
        raise RuntimeError(
            "Unable to load Excel file. Ensure it is a valid .xlsx workbook and that the required engine (openpyxl) is installed. "
            f"Original error: {exc}"
        ) from exc
    finally:
        conn.close()
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def rebuild_metadata(table_name: str) -> None:
    from .text_to_sql_optimized import TextToSQLConfig, TextToSQLSystem

    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        use_gguf=True,
        database_path=str(DUCKDB_PATH),
        vector_store_path=str(VECTOR_STORE_PATH),
        max_correction_attempts=3,
        n_ctx=2048,
        n_threads=0,
        n_gpu_layers=-1,
    )
    system = TextToSQLSystem(config)
    system.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Load Excel into DuckDB")
    parser.add_argument("excel_path", help="Path to the Excel file")
    parser.add_argument("--table", required=True, help="Table name to create")
    parser.add_argument("--sheet", type=int, default=1, help="Sheet index (1-based) to load")
    args = parser.parse_args()

    excel_path = Path(args.excel_path).expanduser().resolve()
    if not excel_path.exists():
        raise SystemExit(f"File not found: {excel_path}")

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Loading {excel_path} into table {args.table}...")
    load_excel_to_duckdb(excel_path, args.table, sheet=args.sheet)
    print("Rebuilding metadata...")
    rebuild_metadata(args.table)
    print("🎉 Done. Restart the web server to use the new table.")


if __name__ == "__main__":
    main()
