"""Custom web application for the Text-to-SQL assistant."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, List
from html import escape
import json
import re
import tempfile

import duckdb
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .text_to_sql_optimized import TextToSQLConfig, TextToSQLSystem
from .load_excel import load_excel_to_duckdb


ROOT = Path(__file__).resolve().parent
STORAGE_ROOT = Path.home() / ".antiks"
DATA_PATH = STORAGE_ROOT / "data"
DUCKDB_PATH = DATA_PATH / "sandbox.duckdb"
VECTOR_STORE_PATH = STORAGE_ROOT / "vector_store"
RELATIONSHIPS_PATH = STORAGE_ROOT / "datasets.json"

SESSION_CONTEXT: Dict[str, Any] = {
    "system": None,
    "last_result": None,
    "last_sql": None,
    "table_keys": {},  # alias -> table name
    "active_alias": None,
    "relationships": [],
    "graph_positions": {},
}

app = FastAPI(title="Text-to-SQL Assistant")

templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def ensure_storage_dirs() -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)


def ensure_data_dir() -> None:
    ensure_storage_dirs()


def cleanup_system() -> None:
    current = SESSION_CONTEXT.get("system")
    if current:
        try:
            current.cleanup()
        except Exception:
            pass
    SESSION_CONTEXT["system"] = None
    get_system.cache_clear()


def load_dataset_state() -> None:
    ensure_data_dir()
    if not RELATIONSHIPS_PATH.exists():
        return
    try:
        data = json.loads(RELATIONSHIPS_PATH.read_text())
    except json.JSONDecodeError:
        return
    SESSION_CONTEXT["table_keys"] = data.get("aliases", {})
    SESSION_CONTEXT["active_alias"] = data.get("active")
    SESSION_CONTEXT["relationships"] = data.get("relationships", [])
    SESSION_CONTEXT["graph_positions"] = data.get("positions", {})


def save_dataset_state() -> None:
    ensure_data_dir()
    payload = {
        "aliases": SESSION_CONTEXT.get("table_keys", {}),
        "active": SESSION_CONTEXT.get("active_alias"),
        "relationships": SESSION_CONTEXT.get("relationships", []),
        "positions": SESSION_CONTEXT.get("graph_positions", {}),
    }
    RELATIONSHIPS_PATH.write_text(json.dumps(payload, indent=2))


load_dataset_state()


class QueryPayload(BaseModel):
    message: str


@lru_cache(maxsize=1)
def get_system() -> TextToSQLSystem:
    ensure_storage_dirs()
    db_path = str(DUCKDB_PATH) if DUCKDB_PATH.exists() else "./sample_database.db"
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        use_gguf=True,
        database_path=db_path,
        vector_store_path=str(VECTOR_STORE_PATH),
        max_correction_attempts=3,
        n_ctx=2048,
        n_threads=0,
        n_gpu_layers=-1,
    )
    system = TextToSQLSystem(config)
    SESSION_CONTEXT["system"] = system
    return system


def reset_system() -> TextToSQLSystem:
    cleanup_system()
    SESSION_CONTEXT["last_result"] = None
    SESSION_CONTEXT["last_sql"] = None
    new_system = get_system()
    SESSION_CONTEXT["system"] = new_system
    return new_system


def register_table_alias(table_name: str) -> str:
    aliases: Dict[str, str] = SESSION_CONTEXT.setdefault("table_keys", {})
    # If table already registered, reuse existing alias
    for alias, existing_table in aliases.items():
        if existing_table == table_name:
            SESSION_CONTEXT["active_alias"] = alias
            return alias
    base_alias = re.sub("[^A-Za-z0-9]", "", table_name.lower())[:3] or "tbl"
    suffix = 1
    alias = base_alias
    while alias in aliases:
        suffix += 1
        alias = f"{base_alias}{suffix}"
    aliases[alias] = table_name
    SESSION_CONTEXT["active_alias"] = alias
    return alias


def extract_where_clause(sql: str) -> Optional[str]:
    lower = sql.lower()
    if " where " not in lower:
        return None
    idx = lower.index(" where ")
    clause = sql[idx:]
    for keyword in [" group by", " order by", " limit", " fetch", ";"]:
        pos = clause.lower().find(keyword)
        if pos != -1:
            clause = clause[:pos]
    return clause.strip()


def augment_message_with_context(message: str) -> str:
    lower = message.lower()
    additions: List[str] = []

    aliases = SESSION_CONTEXT.get("table_keys", {}) or {}
    relationships = SESSION_CONTEXT.get("relationships", []) or []
    active_alias = SESSION_CONTEXT.get("active_alias")

    requested_aliases = {alias for alias in aliases if alias.lower() in lower}
    if not requested_aliases and active_alias and active_alias in aliases:
        requested_aliases.add(active_alias)

    # Expand requested aliases based on declared relationships so related tables become available
    expanded = True
    while expanded:
        expanded = False
        for rel in relationships:
            a1 = rel.get("source_alias")
            a2 = rel.get("target_alias")
            if a1 and a2:
                if a1 in requested_aliases and a2 not in requested_aliases:
                    requested_aliases.add(a2)
                    expanded = True
                elif a2 in requested_aliases and a1 not in requested_aliases:
                    requested_aliases.add(a1)
                    expanded = True

    for alias in sorted(requested_aliases):
        table = aliases.get(alias)
        if not table:
            continue
        additions.append(f"table: {table}")
        additions.append(f"dataset_key {alias} -> {table}")

    if relationships:
        rel_descriptions = []
        for rel in relationships:
            rel_descriptions.append(
                f"{rel.get('source_alias')}({rel.get('source_column')}) {rel.get('join_type', 'inner').upper()} {rel.get('target_alias')}({rel.get('target_column')})"
            )
        if rel_descriptions:
            additions.append("Available dataset joins: " + "; ".join(rel_descriptions))

    context = SESSION_CONTEXT.get("last_result")
    rows = (context or {}).get("rows") or []
    columns = (context or {}).get("columns") or []

    def collect_values(col_name):
        if col_name in columns:
            values = []
            for row in rows:
                val = row.get(col_name)
                if val is None:
                    continue
                val_str = str(val)
                if val_str not in values:
                    values.append(val_str)
            return values
        return []

    last_sql = SESSION_CONTEXT.get("last_sql")
    where_clause = extract_where_clause(last_sql) if last_sql else None

    if rows and columns and "these" in lower:
        product_values = collect_values("product_name") or collect_values("product")
        if product_values and len(product_values) <= 5:
            additions.append(
                "Treat the relevant product_name values as: " + ", ".join(product_values)
            )
        code_values = collect_values("product_code")
        if code_values and len(code_values) <= 5:
            additions.append(
                "Product codes for reference: " + ", ".join(code_values)
            )
        if where_clause:
            additions.append(
                "Reuse the same filters as the previous query instead of enumerating values: " + where_clause
            )

    if additions:
        message = message + "\n" + "\n".join(additions)
    return message


def format_reply(result: Dict) -> str:
    lines = []

    schema_info = result.get("schema_info")
    if schema_info:
        lines.append("<div class=\"section\">")
        lines.append("<p class=\"section__title\">Schema</p>")
        corrections = schema_info.get("corrections") if isinstance(schema_info, dict) else None
        columns = schema_info.get("columns") or []
        if schema_info.get("source") == "query":
            table = schema_info.get("table") or "unknown"
            lines.append(f"<p><strong>Table:</strong> {escape(str(table))}</p>")
        else:
            tables = schema_info.get("tables") or []
            if tables:
                table_list = ", ".join(escape(str(tbl)) for tbl in tables)
                lines.append(f"<p><strong>Tables:</strong> {table_list}</p>")
        if columns:
            cols = ", ".join(escape(str(col)) for col in columns)
            lines.append(f"<p><strong>Columns:</strong> {cols}</p>")
        lines.append("</div>")

        if corrections:
            table_fix = corrections.get("table")
            if table_fix and table_fix.get("normalized") != table_fix.get("original"):
                lines.append(
                    f"<p class=\"schema-note\">Normalized table name from “{escape(str(table_fix.get('original')))}” to “{escape(str(table_fix.get('normalized')))}”.</p>"
                )
            for col_fix in corrections.get("columns", []):
                original = col_fix.get("original")
                normalized = col_fix.get("normalized")
                if original and normalized and original.lower() != normalized.lower():
                    lines.append(
                        f"<p class=\"schema-note\">Matched column “{escape(str(original))}” to “{escape(str(normalized))}”.</p>"
                    )
                elif not original and normalized:
                    lines.append(
                        f"<p class=\"schema-note\">Detected column “{escape(str(normalized))}” from your message.</p>"
                    )

        column_values_map = schema_info.get("column_values_map") or {}
        if column_values_map:
            hints = []
            for col, values in column_values_map.items():
                if not values:
                    continue
                preview = ", ".join(escape(str(v)) for v in values[:4])
                if len(values) > 4:
                    preview += ", …"
                hints.append(f"{col}: {preview}")
            if hints:
                lines.append(f"<p class=\"schema-note\"><strong>Value hints:</strong> {' | '.join(hints)}</p>")

        value_matches = schema_info.get("value_matches") or []
        value_corrections = schema_info.get("value_corrections") or []
        if value_matches:
            parts = [f"{escape(m['column'])} = '{escape(str(m['value']))}'" for m in value_matches]
            lines.append(f"<p class=\"schema-note\">Recognized values from your request: {' | '.join(parts)}</p>")
        if value_corrections:
            parts = [
                f"{escape(c['column'])}: '{escape(str(c['original']))}' → '{escape(str(c['normalized']))}'"
                for c in value_corrections
            ]
            lines.append(f"<p class=\"schema-note\">Adjusted values: {' | '.join(parts)}</p>")

        column_corrections = schema_info.get("column_corrections") or []
        if column_corrections:
            parts = [
                f"{escape(corr.get('requested'))} → {escape(corr.get('table_alias', ''))}.{escape(corr.get('normalized'))}"
                for corr in column_corrections
            ]
            lines.append(f"<p class=\"schema-note\"><strong>Column normalizations:</strong> {' | '.join(parts)}</p>")

        lines.append("</div>")

    sql = result.get("sql") or "-- No SQL generated --"
    lines.append("<div class=\"section\">")
    lines.append("<p class=\"section__title\">SQL</p>")
    lines.append(f"<pre><code>{escape(str(sql))}</code></pre>")
    lines.append("</div>")

    data_columns = result.get("data_columns")
    data_rows = result.get("data_rows")
    if data_columns and data_rows:
        preview = data_rows[:20]
        lines.append("<div class=\"section\">")
        lines.append("<p class=\"section__title\">Result Preview</p>")
        lines.append("<div class=\"table-wrapper\"><table class=\"result-table\">")
        header_cells = "".join(f"<th>{escape(str(col))}</th>" for col in data_columns)
        lines.append(f"<thead><tr>{header_cells}</tr></thead><tbody>")
        for row in preview:
            cells = "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in data_columns)
            lines.append(f"<tr>{cells}</tr>")
        lines.append("</tbody></table></div>")
        total_rows = result.get("row_count")
        if total_rows and total_rows > len(preview):
            lines.append(f"<p class=\"status status--note\">Showing first {len(preview)} rows of {total_rows}.</p>")
        lines.append("</div>")

    status = result.get("status_message") or "SQL generated."
    if result.get("sandbox_validated"):
        badge = "<span class=\"badge badge--success\">Sandbox Verified</span>"
        lines.append(f"<p class=\"status\">{badge} {escape(status)}</p>")
    else:
        lines.append(f"<p class=\"status\">{escape(status)}</p>")

    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query")
async def query(payload: QueryPayload):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    system = get_system()
    message_to_process = augment_message_with_context(message)
    try:
        result = system.query(message_to_process, skip_execution=False, sandbox=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    reply_html = format_reply(result)
    if result.get("data_columns") and result.get("data_rows"):
        SESSION_CONTEXT["last_result"] = {
            "columns": result["data_columns"],
            "rows": result["data_rows"],
        }
    SESSION_CONTEXT["last_sql"] = result.get("sql")
    return JSONResponse({"reply": reply_html})


@app.post("/api/upload")
async def upload_excel(
    file: UploadFile = File(...),
    table: str = Form(...),
    sheet: int = Form(1),
):
    table = table.strip()
    if not table:
        raise HTTPException(status_code=400, detail="Table name is required.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise HTTPException(
            status_code=400,
            detail="Table name must start with a letter or underscore and contain only letters, numbers, or underscores.",
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx or .xls) are supported.")
    if sheet < 1:
        raise HTTPException(status_code=400, detail="Sheet index must be 1 or greater.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_excel_"))
    tmp_path = tmp_dir / (file.filename or "upload.xlsx")

    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)
        cleanup_system()
        load_excel_to_duckdb(tmp_path, table, sheet=sheet)

        conn = duckdb.connect(str(DUCKDB_PATH))
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()

        alias = register_table_alias(table)
        reset_system()
        save_dataset_state()

        aliases = SESSION_CONTEXT.get("table_keys", {})
        dataset_list = [{"alias": k, "table": v} for k, v in aliases.items()]

        return {
            "status": "ok",
            "table": table,
            "rows": row_count,
            "alias": alias,
            "datasets": dataset_list,
            "active": alias,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            tmp_dir.rmdir()
        except OSError:
            pass


@app.get("/api/datasets")
async def list_datasets():
    aliases = SESSION_CONTEXT.get("table_keys", {})
    active_alias = SESSION_CONTEXT.get("active_alias")
    dataset_list = [{"alias": alias, "table": table} for alias, table in aliases.items()]
    return {"datasets": dataset_list, "active": active_alias}


@app.post("/api/refresh")
async def refresh_system():
    reset_system()
    save_dataset_state()
    aliases = SESSION_CONTEXT.get("table_keys", {})
    active_alias = SESSION_CONTEXT.get("active_alias")
    dataset_list = [{"alias": alias, "table": table} for alias, table in aliases.items()]
    return {
        "status": "ok",
        "datasets": dataset_list,
        "active": active_alias,
    }


@app.delete("/api/datasets/{alias}")
async def delete_dataset(alias: str):
    alias = alias.strip()
    aliases = SESSION_CONTEXT.get("table_keys", {})
    table = aliases.get(alias)
    if not table:
        raise HTTPException(status_code=404, detail=f"Unknown dataset alias: {alias}")

    try:
        cleanup_system()
        conn = duckdb.connect(str(DUCKDB_PATH))
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    aliases.pop(alias, None)
    relationships = SESSION_CONTEXT.get("relationships", []) or []
    SESSION_CONTEXT["relationships"] = [
        rel
        for rel in relationships
        if rel.get("source_alias") != alias and rel.get("target_alias") != alias
    ]
    positions = SESSION_CONTEXT.get("graph_positions", {}) or {}
    positions.pop(alias, None)
    if SESSION_CONTEXT.get("active_alias") == alias:
        SESSION_CONTEXT["active_alias"] = None

    save_dataset_state()

    dataset_list = [{"alias": key, "table": value} for key, value in aliases.items()]
    return {
        "status": "ok",
        "datasets": dataset_list,
        "active": SESSION_CONTEXT.get("active_alias"),
    }


@app.get("/api/schema/{alias}")
async def get_schema(alias: str):
    aliases = SESSION_CONTEXT.get("table_keys", {})
    table = aliases.get(alias)
    if not table:
        raise HTTPException(status_code=404, detail=f"Unknown dataset alias: {alias}")
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        finally:
            conn.close()
    except duckdb.Error as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    columns = [row[1] for row in rows]
    return {"alias": alias, "table": table, "columns": columns}


class RelationshipPayload(BaseModel):
    relationships: List[Dict[str, Any]]
    positions: Optional[Dict[str, Any]] = None


@app.get("/api/relationships")
async def get_relationships():
    return {
        "relationships": SESSION_CONTEXT.get("relationships", []),
        "positions": SESSION_CONTEXT.get("graph_positions", {}),
        "active": SESSION_CONTEXT.get("active_alias"),
    }


@app.post("/api/relationships")
async def save_relationships(payload: RelationshipPayload):
    aliases = SESSION_CONTEXT.get("table_keys", {})
    valid_relationships = []
    for rel in payload.relationships or []:
        src_alias = rel.get("source_alias")
        tgt_alias = rel.get("target_alias")
        if not src_alias or not tgt_alias:
            continue
        if src_alias not in aliases or tgt_alias not in aliases:
            continue
        source_column = rel.get("source_column")
        target_column = rel.get("target_column")
        if not source_column or not target_column:
            continue
        joined = {
            "source_alias": src_alias,
            "source_column": source_column,
            "target_alias": tgt_alias,
            "target_column": target_column,
            "join_type": (rel.get("join_type") or "inner").lower(),
        }
        valid_relationships.append(joined)

    SESSION_CONTEXT["relationships"] = valid_relationships
    SESSION_CONTEXT["graph_positions"] = payload.positions or {}
    reset_system()
    save_dataset_state()
    return {
        "status": "ok",
        "relationships": valid_relationships,
        "positions": SESSION_CONTEXT.get("graph_positions", {}),
    }
