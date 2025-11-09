# Multi-Table Text-to-SQL System

## Overview

This system now supports **intelligent multi-table SQL generation** with automatic relationship detection, visual schema management, and advanced query parsing.

## Key Features

### 1. **Automatic Relationship Detection** 🔗

When you upload multiple tables, the system automatically detects potential relationships between them based on:

- **Column name similarity** (e.g., `customer_id` in both tables)
- **Data type compatibility** (both columns are integers or strings)
- **Value overlap** (70%+ of values in one column exist in another)
- **Naming patterns** (foreign key conventions like `fk_*`, `*_id`)

#### Confidence Scoring

Each detected relationship includes:
- **Confidence score** (0-100%): How certain the system is about the relationship
- **Match ratio**: Percentage of overlapping values
- **Reason**: Why this relationship was detected (e.g., "name_match_85% | value_overlap_92%")

### 2. **Visual Schema Board** 📊

The Schema Board provides an interactive interface to:

#### View Relationships
- **All tab**: Shows all relationships (auto-detected + manual)
- **Auto-detected tab**: Shows only relationships found by the AI
- **Manual tab**: Shows only user-created relationships

#### Manage Connections
- **Click columns** to create manual relationships
- **Drag tables** to rearrange the visual layout
- **Choose join types** (INNER, LEFT, RIGHT, FULL)
- **Remove relationships** you don't need

#### Visual Indicators
- **Auto-detected relationships** are highlighted in teal with confidence badges
- **Manual relationships** appear with standard styling
- **Connections are drawn** with curved lines between related columns

### 3. **Intelligent Query Parsing** 🧠

The system analyzes your natural language query to:

#### Identify Required Columns
```
Query: "Show customer names and their total orders"
↓
Detects: customer.name, order.total
```

#### Determine Necessary Tables
- Finds which tables contain the requested columns
- Identifies the minimum set of tables needed

#### Generate Join Paths
- Uses detected relationships to connect tables
- Creates the shortest path between required tables
- Applies appropriate join types

#### Understand Operations
```
Query: "Total sales per customer in 2023"
↓
Operations: SUM, GROUP BY, WHERE (date filter)
```

### 4. **Multi-Table SQL Generation** 📝

The system automatically:

1. **Identifies relationships** between tables mentioned in your query
2. **Generates appropriate JOINs** based on detected or manual relationships
3. **Selects relevant columns** from each table
4. **Applies filters and aggregations** as requested

#### Example Workflow

**User uploads:**
- `customers` table (columns: id, name, email)
- `orders` table (columns: id, customer_id, total, date)

**System auto-detects:**
- Relationship: `orders.customer_id` → `customers.id` (95% confidence)

**User asks:**
- "Show me customer names and their total order amounts"

**System generates:**
```sql
SELECT 
    customers.name,
    SUM(orders.total) AS total_amount
FROM customers
INNER JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.name
ORDER BY total_amount DESC
```

## How to Use

### Step 1: Upload Your Tables

1. Click **"Add table"** in the sidebar
2. Choose an Excel file (.xlsx or .xls)
3. Specify table name and sheet number
4. Click **"Upload"**

The system will:
- Load your data into the sandbox database
- Assign a unique alias (e.g., `cust`, `ord`)
- Auto-detect relationships if multiple tables exist
- Show a summary: "2 relationship(s) auto-detected"

### Step 2: Review Relationships (Optional)

1. Click **"Schema board"** next to any dataset
2. Review auto-detected relationships in the **"Auto-detected"** tab
3. Check confidence scores and reasons
4. Add manual relationships if needed:
   - Click a column in one table
   - Click a column in another table
   - Adjust join type if needed
5. Click **"Save graph"**

### Step 3: Ask Questions

Simply type natural language queries like:

- **Single table**: "Total sales by region"
- **Multi-table**: "Customer names with more than 5 orders"
- **Complex**: "Average order value per customer in 2023, sorted by highest"

The system will:
1. Parse your query
2. Identify required tables and columns
3. Generate appropriate JOINs
4. Execute and show results

### Step 4: Refresh Metadata (When Needed)

Click **"Refresh sandbox"** to:
- Re-detect relationships between all tables
- Update the assistant's knowledge
- Rebuild vector embeddings

## Architecture

### Components

```
┌─────────────────────────────────────┐
│  User Query (Natural Language)      │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│  Multi-Table Query Parser           │
│  - Identifies columns                │
│  - Finds required tables             │
│  - Determines operations             │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│  Relationship Detector               │
│  - Column profiling                  │
│  - Similarity analysis               │
│  - Value overlap detection           │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│  SQL Generation with Joins          │
│  - Build join paths                  │
│  - Apply filters                     │
│  - Add aggregations                  │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│  Execution & Results                 │
└─────────────────────────────────────┘
```

### Data Flow

1. **Upload** → Tables stored in DuckDB
2. **Profile** → Column statistics and sample values extracted
3. **Detect** → Relationships identified automatically
4. **Store** → Metadata saved to `~/.antiks/datasets.json`
5. **Query** → Natural language parsed into structured intent
6. **Generate** → SQL with JOINs created
7. **Execute** → Results returned to user

## Technical Details

### Relationship Detection Algorithm

```python
confidence = 0.0

# Name similarity (30% weight)
if column_names_similar > 70%:
    confidence += 0.3 * similarity_ratio

# Value overlap (50% weight)
if value_overlap > 70%:
    confidence += 0.5 * overlap_ratio

# Key indicators (15% weight)
if target_is_unique_key:
    confidence += 0.15

# Primary key (5% weight)
if target_is_primary_key:
    confidence += 0.05

# FK naming pattern (10% weight)
if follows_fk_convention:
    confidence += 0.10
```

### Query Parsing Logic

1. **Tokenize** query into words
2. **Match tokens** to column names (exact and fuzzy)
3. **Match values** to detect filter columns
4. **Identify operations** (SUM, AVG, GROUP BY, etc.)
5. **Determine required tables** from matched columns
6. **Build join path** using relationships
7. **Generate structured intent** for SQL creation

## Best Practices

### Naming Conventions

For best auto-detection results, use consistent naming:

- **Primary keys**: `id`, `customer_id`, `product_code`
- **Foreign keys**: `customer_id`, `product_id`, `order_id`
- **Timestamps**: `created_at`, `updated_at`, `order_date`

### Manual Relationships

Create manual relationships when:

- Auto-detection confidence is low (<60%)
- Tables use unconventional naming
- You want to override auto-detected relationships
- Complex business logic requires specific joins

### Query Tips

For best results:

✅ **Good**: "Show customer names and their total order amounts"
✅ **Good**: "Average sales per region in 2023"
✅ **Good**: "Top 10 customers by revenue"

❌ **Avoid**: "Show me stuff from customers and orders" (too vague)
❌ **Avoid**: Using table names without context (system will auto-detect)

## Configuration

### Adjust Detection Thresholds

In `relationship_detector.py`:

```python
NAME_SIMILARITY_THRESHOLD = 0.7  # 70% name match required
VALUE_MATCH_THRESHOLD = 0.7      # 70% value overlap required
MIN_CONFIDENCE = 0.5             # 50% minimum confidence
```

### Storage Locations

- **Sandbox database**: `~/.antiks/data/sandbox.duckdb`
- **Relationships**: `~/.antiks/datasets.json`
- **Vector store**: `~/.antiks/vector_store/`

## API Endpoints

### Auto-detect Relationships
```
POST /api/refresh
Response: { auto_detected_relationships: number }
```

### Get Schema
```
GET /api/schema/{alias}
Response: { alias, table, columns[] }
```

### Get/Save Relationships
```
GET /api/relationships
Response: { relationships[], auto_detected_relationships[], positions{} }

POST /api/relationships
Body: { relationships[], positions{} }
```

## Troubleshooting

### No Relationships Detected

**Issue**: System doesn't find relationships between tables

**Solutions**:
1. Check if column names are similar
2. Verify data types are compatible
3. Ensure some values overlap between columns
4. Create manual relationships in Schema Board

### Wrong Join Generated

**Issue**: SQL uses incorrect join

**Solutions**:
1. Specify join type in Schema Board
2. Be more specific in your query about which tables
3. Use "Refresh sandbox" to rebuild metadata

### Query Uses Wrong Tables

**Issue**: System selects unexpected tables

**Solutions**:
1. Be explicit: "from customers table, show..."
2. Mention specific columns: "customer name and order total"
3. Check Schema Board for unwanted relationships

## Future Enhancements

Potential improvements:

- [ ] Multi-level joins (A→B→C)
- [ ] Self-joins detection
- [ ] Relationship strength visualization
- [ ] Query history with join patterns
- [ ] Suggested relationships based on query patterns
- [ ] Export schema diagrams

## Support

For issues or questions:
1. Check the Schema Board for relationship status
2. Review auto-detected confidence scores
3. Try manual relationships if auto-detection fails
4. Use "Refresh sandbox" to rebuild metadata

