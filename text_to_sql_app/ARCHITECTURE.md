# Multi-Table Text-to-SQL Architecture

This document explains, end-to-end, how the system turns a user's natural language request into safe, executable SQL. It aims to be the canonical reference for future contributors who need to understand the architecture or extend it for new edge-cases.

---

## 1. System Overview

```
┌────────────────────┐
│   FastAPI Server   │  /api/query
└────────┬───────────┘
         │ message payload
         ▼
┌─────────────────────────────────────────────────┐
│ TextToSQLSystem (text_to_sql_optimized.py)      │
│                                                 │
│ 1. Load metadata (tables, columns, sample values│
│ 2. Detect columns / parse query intent          │
│ 3. Build LLM prompt (single / multi-table)      │
│ 4. Call LLM (defog/sqlcoder-7b-2)               │
│ 5. Sanitize SQL (joins/tables/columns/literals) │
│ 6. Execute via DuckDB                           │
└───────┬─────────────────────────────────────────┘
        │ result + metadata
        ▼
┌────────────────────┐       ┌─────────────────────┐
│   FastAPI Server   │──────▶│  Browser Frontend   │
└────────────────────┘       └─────────────────────┘
```

- The **FastAPI server** (`text_to_sql_app/server.py`) exposes `/api/query`. It collects metadata, orchestrates the query, and post-processes the reply into HTML.
- The **TextToSQLSystem** coordinates column detection, prompt construction, SQL generation, validation, and execution.
- The **LLM** (`defog/sqlcoder-7b-2` in GGUF format) generates candidate SQL based on the carefully crafted prompts.
- The **DuckDB sandbox** stores user-uploaded tables (via `/api/upload`) and executes the final SQL.

---

## 2. Metadata Loading & Session Context

Metadata is loaded from the current DuckDB database via `TextToSQLSystem._initialize_system()`:

1. Retrieves all tables, columns, types, and sample distinct values (`_load_table_columns()`).
2. Populates `self.table_columns_map`, `self.relationships`, and `self.column_embeddings`.
3. Metadata is cached in `SESSION_CONTEXT`:
   - `tables`, `columns`, `relationships`
   - `user_defined_relationships`
   - `auto_detected_relationships`

Sidecar modules:
- `relationship_detector.py`: finds potential joins via column name similarity and value overlap.
- `multi_table_query_parser.py`: interprets the NL query to identify required tables, columns, joins, and operations.
- `column_alternatives.py`: suggests top alternative columns per query token.

---

## 3. Frontend Flow (Browser)

1. User types a question; `sendMessage()` posts to `/api/query`.
2. While waiting, a “Thinking…” bubble is shown.
3. After the response:
   - Detected columns (with scores)
   - Column alternatives (chips per table alias)
   - SQL and any error/result sections
4. Clicking an alternative chip re-issues the query with a hint (e.g. `use adh.minority for "minority"`).
5. Sidebar (desktop) or hamburger drawer (mobile) lets users upload datasets, refresh metadata, or explore relationships.

UI assets:
- `templates/index.html`
- `static/style.css`
- WhatsApp-style composer with auto-expanding textarea.

---

## 4. Request Handling (`/api/query`)

1. Validate payload: non-empty `message`.
2. Augment message with context (last SQL, last result, serializer hints).
3. Call `system.query(message_to_process, skip_execution=False, sandbox=True)`.
4. Catch exceptions and return HTML with structured error details.
5. On success:
   - Cache `last_sql`, `last_result`.
   - Format the HTML output (detected columns, column alternatives, SQL, execution time, result table or chart).

---

## 5. TextToSQLSystem.query(...) Pipeline

`TextToSQLSystem.query(...)` handles everything from metadata usage to final execution.

### 5.1 Preprocessing
- Loads metadata if not already loaded.
- Determines `allowed_tables` based on instruction set (CSV, pivot, fallback, etc.).
- Optionally refines query (remove appended schema hints, clarify aggregate intent).

### 5.2 Column Detection & Alternatives
- `_detect_columns_from_query()`: tokenizes query, matches tokens to columns (`exact`, `substring`, `fuzzy`).
- For multi-table scenarios, `parse_multi_table_query()` builds a `QueryIntent`:
  - `requested_columns`, `required_tables`, `required_joins`, `operations`, `filters`.
- Detected columns stored in `schema_info_used["column_detections"]`.
- `group_alternatives_by_query_term()` computes top 3 alternatives per token; passed to frontend for chip rendering.

### 5.3 Prompt Construction
Two branches:

1. **Single-table** → `_build_prompt(...)`
   - explicit warning: “Use ONLY columns listed below”
   - includes `### AVAILABLE COLUMNS` list (table alias → columns)
   - analysis block summarizing operations, columns, tables
2. **Multi-table** → `_build_multi_table_prompt(...)`
   - same warnings + join instructions
   - explicit join list e.g. `INNER JOIN chg ON adh.loan_account_id = chg.loan_account_id`
   - includes aggregated `AVAILABLE COLUMNS` per alias

Both prompts emphasize:
- No invented columns/tables
- Pre-aggregate joined tables in a CTE before joining so each base row appears once
- Follow join specs
- Group by non-aggregates
- No extra filters
- Return SQL only

### 5.4 LLM Call & Correction Loop

`max_correction_attempts` (default 3). Steps:

1. Call LLM with prompt → SQL candidate.
2. Sanitize & validate (see Section 6).
3. If validation fails, append error to correction instructions and retry.

---

## 6. SQL Sanitization & Validation

Performed in strict order to guarantee schema compliance:

1. **_rewrite_unsupported_functions**  
   - Converts `TO_DATE(expr, '...')` → `TRY_CAST(expr AS DATE)` and `TO_CHAR(expr, '...')` → `CAST(expr AS VARCHAR)` so format tokens don’t get treated as columns.  
2. **_sanitize_sql_tables**  
   - Remove JOINs referencing unloaded tables.  
   - Replace `FROM` table with allowed one if mismatched.

3. **_sanitize_sql_columns**  
   - Build `allowed_columns` per alias from metadata/prompt.  
   - `choose_column()` resolves each column reference using:
     - exact match → normalized match → token map → fuzzy (0.75 / 0.65 / 0.55) → fallback column.
   - Logs warning/error for replacements.

4. **_normalize_sql_literals**  
   - Snap string literals to closest known values (from stored distinct values, config-supplied map).  
   - Handles synonyms (yes/no/true/false/on/off) → `'Y'/'N'`.
   - Works for equality and `IN (...)`.

5. **_coerce_numeric_comparisons**  
   - Wrap numeric comparisons (`>`, `<`, `>=`, `<=`, `=`) with `TRY_CAST(column AS DOUBLE)` and handle quoted numerics.

6. **_rewrite_joined_aggregates**  
   - Detects when the primary table is aggregated while another table is joined. If so, raises a `ValueError` with guidance to pre-aggregate the joined table into a summary CTE keyed by the join columns, preventing duplicate counting.

7. **_coerce_numeric_operations**  
   - Cast all arithmetic (`-`, `+`, `*`, `/`) operands to `DOUBLE`.

8. **_cast_numeric_aggregates**  
   - Additional safety for aggregates (e.g., `SUM(TRY_CAST(...))` if needed).

9. **_validate_sql_columns**  
   - Cross-check final SQL against allowed tables/columns.  
   - If mismatch persists, raise `ValueError` → triggers correction loop.

These layers ensure SQL matches the actual schema even if the LLM attempts to hallucinate.

---

## 7. Filter Value Normalization

When parsing filters or sanitizing SQL:
- Distinct values retrieved during metadata loading (limit 40 per column).
- `_normalize_sql_literals` uses:
  - direct case-insensitive match
  - synonyms (`yes`→`Y`, `true`→`Y`, etc.)
  - fuzzy match via `difflib.get_close_matches`
  - sequence match fallback (score ≥ 0.6)
- If no match, literal remains unchanged (so user can discover invalid values).

Example:
```
WHERE a.is_priority = 'yes'
→ WHERE a.is_priority = 'Y'
```

---

## 8. Column Alternatives & UI Chips

`group_alternatives_by_query_term()` produces:

```json
{
  "priority": [
    {"table_alias": "adh", "column_name": "is_priority", "similarity": 0.92, "is_selected": true},
    {"table_alias": "adh", "column_name": "priority_flag", "similarity": 0.88, "is_selected": false}
  ]
}
```

- Frontend renders table-grouped chips (teal pill buttons).
- Clicking a chip re-issues query with hint `"... (use adh.priority_flag for \"priority\")"`.
- Selected column is disabled; others highlight on hover.

---

## 9. Relationship Detection & Schema Board

Relationships combine:

1. **Auto-detected** (`relationship_detector.py`)  
   - Column name similarity (`loan_account_id` vs `loan_account_id`).  
   - Value overlap (Jaccard > threshold).
   - Type compatibility (INT with INT, etc.).
   - Direction suggestion (foreign key vs primary key).

2. **User-defined** (via schema board UI).

`SESSION_CONTEXT["relationships"]` merges both. Graph board (HTML overlay) includes nodes per table with draggable positions; connections show join columns. Tabbed interface toggles auto vs manual edges.

---

## 10. Edge-Case Handling Summary

| Edge Case | Mitigation |
|-----------|------------|
| **Missing table** | `_sanitize_sql_tables` removes disallowed JOINs; fallback to allowed table. |
| **Missing column** | `_sanitize_sql_columns` aggressively rewrites references to closest valid column; logs error/warning. |
| **String vs numeric comparison** | `_coerce_numeric_comparisons` wraps column in `TRY_CAST(... AS DOUBLE)`. |
| **String literal mismatch** | `_normalize_sql_literals` snaps to known values (synonyms, fuzzy match). |
| **Arithmetic on strings** | `_coerce_numeric_operations` casts operands to `DOUBLE`. |
| **Multi-table join instructions** | `parse_multi_table_query` ensures join paths exist in relationships; prompt includes explicit join list. |
| **Alternative column suggestions** | `column_alternatives.py` surfaces top candidates in UI. |
| **Single-table misuse** | Prompt explicitly forbids JOINs; `_sanitize_sql_tables` strips them. |
| **Pivot/grouping hints** | `_build_prompt` adds special instructions for pivots, fallback queries, or corrections. |
| **LLM hallucinations** | Multiple prompt warnings + post-sanitization ensures final SQL matches schema. |

---

## 11. Configuration & Environment

- Python 3.11
- Dependencies in `requirements_text_to_sql.txt` (includes `nltk`, `duckdb`, `fastapi`, `chromadb`, `sentence-transformers`, `pandas`, etc.)
- LLM model: `defog/sqlcoder-7b-2` (GGUF) loaded via custom runner.
- DuckDB sandbox file: `~/.antiks/data/sandbox.duckdb`
- Metadata cache: `~/.antiks/datasets.json`
- Backend entrypoint: `uvicorn text_to_sql_app.server:app --reload`

---

## 12. Workflow Checklist for Contributors

1. **Upload test tables** via UI or script; click “Refresh sandbox”.
2. **Run queries** in UI; observe detected columns, alternatives, SQL, and results.
3. **Check logs** (console) for warning/error messages (e.g., fuzzy matches, literal normalization).
4. **Update sanitization logic** if new edge cases appear (e.g., date comparisons, boolean casting).
5. **Extend `PROMPT_RULES.md`** to reflect behavioral changes.
6. **Add unit tests** (see `text_to_sql_app/test_*`) for new scenarios.
7. **Document** new features and edge-case mitigations in this file.

---

## 13. Future Enhancements (Ideas)

- **Date literal normalization** (e.g., map “today”, “prev month” to actual dates using metadata).
- **Boolean coercion** beyond Y/N (true, false, 1, 0).
- **Join path ranking** when multiple relationships exist; automatically choose the highest-confidence chain.
- **Explainable rewrites** exposed to the user (why a column/literal was normalized).
- **LLM fallback** to smaller model, or self-consistency (multiple generations).

---

Maintaining this document ensures we have a single source of truth about the Text-to-SQL architecture. When integrating new features or handling new edge cases, update the relevant section so future engineers understand both the intent and the implementation.

