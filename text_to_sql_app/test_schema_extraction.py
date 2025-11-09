"""Quick test for schema extraction"""
import re

def extract_schema_from_query(user_query: str):
    """Extract table and column information from user query."""
    schema_info = {
        "table_name": None,
        "columns": []
    }
    
    # Extract table name
    table_patterns = [
        r"table\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"table\s*[:=]\s*([a-zA-Z_][a-zA-Z0-9_]*)",
        r"and\s+table\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    ]
    
    for pattern in table_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            schema_info["table_name"] = match.group(1)
            print(f"✓ Extracted table: {schema_info['table_name']}")
            break
    
    # Extract columns
    column_patterns = [
        r"given\s+columns?\s+are\s+([^.\n]+?)(?:\s+and\s+table|$)",
        r"columns?\s+are\s+([^.\n]+?)(?:\s+and\s+table|$)",
    ]
    
    for pattern in column_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            columns_str = match.group(1).strip()
            print(f"  Raw columns string: '{columns_str}'")
            columns = [col.strip() for col in re.split(r'[,;]', columns_str)]
            columns = [col for col in columns if col and col.lower() not in ['and', 'table']]
            schema_info["columns"] = columns
            print(f"✓ Extracted columns: {schema_info['columns']}")
            break
    
    return schema_info

# Test with user's query
test_query = "can you write a query to get the disbursement amount zone wise. given columns are disbursement_amount,Zone_Name and table is disbursement_register"

print("Testing schema extraction:")
print(f"Query: {test_query}\n")
result = extract_schema_from_query(test_query)

print(f"\nResult: {result}")
print(f"\nTable: {result['table_name']}")
print(f"Columns: {result['columns']}")

