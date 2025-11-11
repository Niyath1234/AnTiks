# Quick Start Guide - Multi-Table Text-to-SQL

## Setup

1. **Start the server**:
```bash
cd /Users/niyathnair/.cursor/worktrees/AnTiks/BeFcX
source text_to_sql_env/bin/activate
python -m text_to_sql_app.server
```

2. **Open browser**: http://localhost:8000

## Test the New Features

### Test 1: Auto-Detect Relationships

1. **Create test Excel files**:

**customers.xlsx**:
| id | name        | email              |
|----|-------------|--------------------|
| 1  | Alice Smith | alice@example.com  |
| 2  | Bob Jones   | bob@example.com    |
| 3  | Carol White | carol@example.com  |

**orders.xlsx**:
| id | customer_id | product      | amount |
|----|-------------|--------------|--------|
| 1  | 1           | Laptop       | 1200   |
| 2  | 1           | Mouse        | 25     |
| 3  | 2           | Keyboard     | 75     |
| 4  | 3           | Monitor      | 300    |

2. **Upload both files**:
   - Upload `customers.xlsx` → table name: `customers`
   - Upload `orders.xlsx` → table name: `orders`
   - Watch for message: "Found X potential relationship(s)"

3. **Check Schema Board**:
   - Click "Schema board" next to any dataset
   - Should see connection: `orders.customer_id` → `customers.id`
   - Check "Auto-detected" tab to see confidence score

### Test 2: Multi-Table Query

After uploading the test data, try these queries:

```
"Show customer names with their total order amounts"
```

Expected SQL:
```sql
SELECT 
    customers.name,
    SUM(orders.amount) AS total_amount
FROM customers
INNER JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.name
```

```
"List all customers and how many orders they placed"
```

Expected SQL:
```sql
SELECT 
    customers.name,
    COUNT(orders.id) AS order_count
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.name
```

```
"Who bought a laptop?"
```

Expected SQL:
```sql
SELECT DISTINCT customers.name
FROM customers
INNER JOIN orders ON customers.id = orders.customer_id
WHERE orders.product = 'Laptop'
```

### Test 3: Manual Relationships

1. **Upload unrelated table**:

**products.xlsx**:
| code | name     | price |
|------|----------|-------|
| L01  | Laptop   | 1200  |
| M01  | Mouse    | 25    |
| K01  | Keyboard | 75    |

2. **Create manual relationship**:
   - Open Schema Board
   - Click `orders.product` column
   - Click `products.name` column
   - Choose join type: INNER
   - Save

3. **Query across three tables**:
```
"Show customer names and product prices"
```

Should join: customers → orders → products

### Test 4: Relationship Management

1. **View tabs**:
   - Click "All" tab: See all relationships
   - Click "Auto-detected" tab: See only AI-found
   - Click "Manual" tab: See only user-created

2. **Modify relationships**:
   - Change join type (INNER → LEFT)
   - Remove unwanted relationships
   - Save changes

3. **Refresh**:
   - Click "Refresh sandbox"
   - Re-detects relationships
   - Merges with your manual ones

## Verify Features

### ✅ Auto-Detection Works
- Upload 2+ related tables
- Check notification: "X relationship(s) auto-detected"
- Open Schema Board
- See connection with confidence badge

### ✅ Schema Board Works
- Tables are draggable
- Tabs switch correctly
- Confidence badges show on auto-detected
- Manual connections can be created
- Join types are adjustable

### ✅ Multi-Table Queries Work
- Ask about multiple tables
- SQL includes JOINs
- Correct columns selected
- Results make sense

### ✅ Manual Override Works
- Create relationship manually
- Query uses that relationship
- Can override auto-detected ones

## Common Issues

### No Relationships Detected

**Symptom**: Upload 2 tables, no relationships found

**Fixes**:
- Check column names are similar (`customer_id` vs `id`)
- Verify some values overlap
- Try manual relationship in Schema Board

### Wrong Join Generated

**Symptom**: Query produces incorrect results

**Fixes**:
- Check Schema Board for relationship
- Verify join type is correct
- Try being more specific in query

### Tables Not Connecting

**Symptom**: Query says "no relationships found"

**Fixes**:
- Click "Refresh sandbox"
- Check Schema Board shows connection
- Create manual relationship if needed

## Advanced Testing

### 3-Table Join

Upload:
- `customers` → `orders` → `products`

Query:
```
"Total sales per customer for laptops"
```

Should join all three tables with appropriate filters.

### Aggregation + Filter

Query:
```
"Customers who spent more than $500"
```

Should:
1. JOIN tables
2. SUM order amounts
3. GROUP BY customer
4. HAVING SUM > 500

### Multiple Relationships

If tables have multiple possible joins:
- System picks highest confidence
- You can override in Schema Board
- Query can specify which to use

## Performance Benchmarks

Expected performance:
- Upload + detection: 2-5 seconds
- Query parsing: <100ms
- SQL generation: 2-3 seconds
- Total query time: 3-5 seconds

## Next Steps

1. **Test with your real data**
2. **Review auto-detected relationships** (some may need manual override)
3. **Save common queries** for reuse
4. **Report any issues** or unexpected behavior

## Example Session

```
1. Upload customers.xlsx → "customers" table
   ✅ Loaded 100 rows

2. Upload orders.xlsx → "orders" table
   ✅ Loaded 500 rows
   🎯 Found 1 relationship: orders.customer_id → customers.id (92%)

3. Open Schema Board
   👀 See auto-detected connection
   ⭐ Confidence: 92%
   📝 Reason: name_match_85% | value_overlap_98%

4. Ask: "Top 10 customers by total orders"
   🤖 Parsing query...
   📊 Tables: customers, orders
   🔗 Join: customers.id = orders.customer_id
   💡 Operation: SUM + GROUP BY + ORDER BY + LIMIT
   
   ✅ Generated SQL:
   SELECT customers.name, SUM(orders.amount) as total
   FROM customers
   INNER JOIN orders ON customers.id = orders.customer_id
   GROUP BY customers.name
   ORDER BY total DESC
   LIMIT 10

5. See results with customer names and totals
```

## Support

If you encounter issues:

1. Check browser console for errors
2. Check server logs
3. Verify data in Schema Board
4. Try "Refresh sandbox"
5. Review `~/.antiks/datasets.json`

Enjoy your new multi-table SQL capabilities! 🚀

