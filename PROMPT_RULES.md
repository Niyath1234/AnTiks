# Prompt Rules for SQL Generation

The Text-to-SQL assistant follows these hard constraints every time it builds a query. The backend enforces them even if the language model tries to deviate, so anything not listed here is rejected or auto-corrected before execution.

## 1. Tables
- **Only use tables that are currently loaded in the sandbox.**  
  Any table name not present in `~/.antiks/datasets.json` is removed from the SQL.
- **Strip stale joins automatically.**  
  If a generated query references a non-existent table (e.g., `product`), the JOIN clause is dropped before validation.
- **Single-table queries stay single-table.**  
  When all requested columns and filters come from one table, no JOIN is permitted; any extra joins are removed.

## 2. Columns
- **Exact column names only.**  
  Every column in the final SQL must exist in the loaded schema. The sanitizer aggressively rewrites hallucinated columns to the highest-confidence real column (e.g., `loan_id` → `loan_account_id`). If no match is found, it falls back to the first available column and logs a warning.
- **Best-match selection.**  
  The parser ranks candidate columns using exact, substring, and fuzzy matching (cutoff=0.75 for strict, 0.65 for medium, 0.55 for sequence ratio); the top score is used automatically and alternatives are exposed in the UI.
- **Clickable alternatives.**  
  When multiple columns have similar scores, the assistant displays up to three alternatives. Clicking a chip re-runs the query with that column swapped in.
- **Filter value normalization.**  
  When a filter uses a column with a small value domain (<40 distinct values), the system automatically snaps the literal to the closest valid value (e.g., `'yes'` → `'Y'`). Synonyms like `yes/no`, `true/false`, `on/off` are mapped to common single-character codes.
- **Automatic numeric casting.**  
  All numeric comparisons (>, <, >=, <=, =) automatically cast VARCHAR columns to DOUBLE (e.g., `c.Charge > 200` becomes `TRY_CAST(c.Charge AS DOUBLE) > 200`).
- **Automatic arithmetic casting.**  
  Arithmetic operations between columns (-, +, *, /) automatically cast both operands to DOUBLE to prevent type errors.

## 3. Joins
- **Only join tables whose columns are required.**  
  The assistant computes the minimal table set needed for the requested columns/filters and limits joins to that set.
- **Join specs obey relationships.**  
  Join paths come from the current relationship graph (manual + auto-detected). If a join path is missing, the assistant won’t invent one.

## 4. Prompt instructions sent to the LLM
Every prompt includes:
- The exact schema context with table names, column names, data types, and sample values.
- **A dedicated "AVAILABLE COLUMNS" section** that exhaustively lists every column by table alias, making it impossible to claim ignorance.
- Explicit warnings: "CRITICAL: Use ONLY columns listed below. DO NOT invent, guess, or use similar column names. If a concept is not in the available columns, you CANNOT include it in the SQL."
- Rules such as "Follow join specifications exactly," "Group by all non-aggregated columns," and "Return only the final SQL query without explanations."

## 5. Post-generation sanitation
After the model returns SQL, the backend applies these sanitization steps in order:
1. **Remove disallowed tables/joins** – any table not in the current sandbox is stripped from the query.
2. **Rewrite column names** – every column reference is matched against the schema using exact match → normalized match → token map → fuzzy match (cutoff=0.75, 0.65, 0.55). If no match is found, it falls back to the first available column and logs an error.
3. **Normalize filter literals** – string literals in WHERE clauses are snapped to the closest known value from the column's distinct values (e.g., `'yes'` → `'Y'`).
4. **Cast numeric comparisons** – all comparisons with numeric literals automatically wrap the column in `TRY_CAST(... AS DOUBLE)`.
5. **Cast arithmetic operations** – subtraction, addition, multiplication, and division between columns wrap both operands in `TRY_CAST(... AS DOUBLE)`.
6. **Validate schema** – confirm no missing tables/columns remain.
7. **Execute** – only if all validations pass; otherwise, the correction loop restarts with the error message and sanitized SQL.

## 6. Error handling
- Any failure prints the exact error and stack trace in the UI.
- The system automatically retries up to `max_correction_attempts` (default 3) with the sanitized SQL and validation feedback.

These rules ensure generated SQL is always consistent with the actual sandbox schema and minimize manual intervention. Update this document whenever the prompting or sanitization logic changes.

