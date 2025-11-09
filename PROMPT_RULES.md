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
  Every column in the final SQL must exist in the loaded schema. We rewrite hallucinated columns to the highest-confidence real column (e.g., `loan_id` → `loan_account_id`).
- **Best-match selection.**  
  The parser ranks candidate columns using exact, substring, and fuzzy matching; the top score is used automatically and alternatives are exposed in the UI.
- **Clickable alternatives.**  
  When multiple columns have similar scores, the assistant displays up to three alternatives. Clicking a chip re-runs the query with that column swapped in.
- **Filter normalization.**  
  (In progress) When a filter uses a column with a small value domain (<40 distinct values), the system snaps the literal to the closest valid value.

## 3. Joins
- **Only join tables whose columns are required.**  
  The assistant computes the minimal table set needed for the requested columns/filters and limits joins to that set.
- **Join specs obey relationships.**  
  Join paths come from the current relationship graph (manual + auto-detected). If a join path is missing, the assistant won’t invent one.

## 4. Prompt instructions sent to the LLM
Every prompt includes:
- The exact schema context with table names, column names, and sample values.
- An explicit list of the only column names allowed.
- Rules such as “Use ONLY the exact table and column names provided,” “Do not guess columns,” and “Return only the final SQL.”

## 5. Post-generation sanitation
After the model returns SQL, the backend:
1. Removes disallowed tables/joins.
2. Rewrites column names to real ones.
3. Validates the schema (no missing tables/columns).
4. Applies numeric casting guardrails.
5. Executes only if validation passes; otherwise, the correction loop restarts with the error message.

## 6. Error handling
- Any failure prints the exact error and stack trace in the UI.
- The system automatically retries up to `max_correction_attempts` (default 3) with the sanitized SQL and validation feedback.

These rules ensure generated SQL is always consistent with the actual sandbox schema and minimize manual intervention. Update this document whenever the prompting or sanitization logic changes.

