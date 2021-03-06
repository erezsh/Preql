# Code comparison: Preql, SQL and the rest

This document was written with the aid of [Pandas' comparison with SQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html).

Use the checkboxes to hide/show the code examples for each language.

<script src="https://cdnjs.cloudflare.com/ajax/libs/cash/8.1.0/cash.min.js"></script>

<div class="sticky">
	<form>
		<div id="dialect_choices">
			<div class="choice">
				<input type="checkbox" id="preql" onclick="$('.preql').toggle()" checked="checked">
				<label for="preql">Preql</label>
			</div>
			<div class="choice">
				<input type="checkbox" id="sql" onclick="$('.sql').toggle()" checked="checked">
				<label for="sql">SQL</label>
			</div>
			<div class="choice">
				<input type="checkbox" id="pandas" onclick="$('.pandas').toggle()" checked="checked">
				<label for="pandas">Pandas</label>
			</div>
			<div class="choice">
				<input type="checkbox" id="sqlalchemy" onclick="$('.sqlalchemy').toggle()" checked="checked">
				<label for="sqlalchemy">SQLAlchemy</label>
			</div>
		</div>
	</form>
</div>

<style>
	.preql::before {
	  content: "Preql ";
	}
	.sql::before {
	  content: "SQL ";
	}
	.pandas::before {
	  content: "Pandas ";
	}
	.sqlalchemy::before {
	  content: "SQLAlchemy ";
	}
	.preql::before, .sql::before, .pandas::before, .sqlalchemy::before {
	  font-weight: bold;
	  font-size: 1.1em;
	  color: #489;
	  margin-bottom: 5px;
	  text-decoration: underline;
	  line-height: 2em;
	}
	div.sticky {
		position: -webkit-sticky;
		position: sticky;
		top: 0;
	}
	.choice {
		display: flex;
		padding: 10px;
	}
	#dialect_choices {
		background: white;
		padding: 20px;
		display: flex;
	}
</style> 

## Table Operations

### Selecting columns


<div class="preql">

Column selection is done using the projection operator, `{}`.

```javascript
tips{total_bill, tip, smoker, time}
```

The table name (`tips`) comes first, so that Preql can automatically suggest the field names.

</div>

<div class="sql">

In SQL, selection is done using the SELECT statement

```SQL
SELECT total_bill, tip, smoker, time FROM tips;
```
</div>

<div class="pandas">

```python
tips[['total_bill', 'tip', 'smoker', 'time']]
```

</div>


### Filtering rows

<div class="preql">

Row filtering is done using the filter operator, `[]`:

```javascript
tips[size >= 5 or total_bill > 45]
```
</div>

<div class="sql">

```SQL
SELECT * FROM tips WHERE size >= 5 OR total_bill > 45;
```
</div>

<div class="pandas">

DataFrames can be filtered in multiple ways; Pandas suggest using boolean indexing:

```python
tips[(tips['size'] >= 5) | (tips['total_bill'] > 45)]
```
</div>

<div class="sqlalchemy">

```python
from sqlalchemy import or_
session.query(Tips).filter(or_(Tips.size >= 5, Tips.total_bill > 45))
```

</div>

### Group by / Aggregation

In this example, we calculate how the amount of tips differs by day of the week.

<div class="preql">

Preql extends the projection operator to allow aggregation using the `=>` construct:

```javascript
tips{day => avg(tip), count()}
```

Conceptually, everything on the left of `=>` are the keys, and on the right are the aggregated values.

</div>

<div class="sql">

```SQL
SELECT day, AVG(tip), COUNT(*) FROM tips GROUP BY day;
```
</div>

<div class="pandas">

```python
tips.groupby('day').agg({'tip': np.mean, 'day': np.size})
```

</div>

<div class="sqlalchemy">

```python
from sqlalchemy import func
session.query(Tips.day, func.avg(Tips.tip), func.count(Tips.id)).group_by(Tips.day).all()
```
</div>

### Concat, Union

In this example, we will concatenate and union two tables together.

<div class="preql">

```javascript
table1 + table2	// concat
```

```javascript
table1 | table2	// union
```

</div>

<div class="sql">

```SQL
SELECT * FROM table1 UNION ALL SELECT * FROM table2;  -- concat
```

```SQL
SELECT * FROM table1 UNION SELECT * FROM table2;      -- union
```
</div>

<div class="pandas">

```python
pd.concat([table1, table2])                      # concat
```

```python
pd.concat([table1, table2]).drop_duplicates()    # union
```
</div>


<div class="sqlalchemy">

```python
union_all(session.query(table1), session.query(table2))      # concat
```

```python
union(session.query(table1), session.query(table2))          # union
```
</div>

### Top n rows with offset (limit)

<div class="preql">

```javascript
tips[5..15]
// OR
tips[5..][..10]
```
</div>

<div class="sql">

```SQL
SELECT * FROM tips ORDER BY tip DESC LIMIT 10 OFFSET 5;
```
</div>

<div class="pandas">

```python
tips.nlargest(10 + 5).tail(10)
```
</div>



### Join

Join is essentially an operation that matches rows between two tables, based on common attributes.

<div class="preql">

```javascript
join(a: table1.key1, b: table2.key2)
```

The result is a table with two columns, `a` and `b`, which are structs that each contain the columns of their respective table.

If we have pre-defined a "default join" between tables, we can shorten it to:

```javascript
join(a: table1, b: table2)
```

Preql also offers the functions `leftjoin()`, `outerjoin()`, and `joinall()`.
</div>


<div class="sql">

```SQL
SELECT * FROM table1 INNER JOIN table2 ON table1.key1 = table2.key2;
```
</div>

<div class="pandas">

```python
pd.merge(df1, df2, on='key')
```

(it gets complicated if the key isn't with the same name)
</div>


<div class="sqlalchemy">

```python
session.query(Table1).join(Tables2).filter(Table1.key1 == Table2.key2)
```

</div>

### Insert row

Insert a row to the table, and specifying the columns by name.

<div class="preql">

```javascript
new Country(name: "Spain", language: "Spanish")
```
</div>

<div class="sql">

```sql
INSERT INTO Country (name, language) VALUES ("Spain", "Spanish")
```

</div>

<div class="pandas">

```python
countries = countries.append({'name':'Spain', 'language': 'Spanish'}, ignore_index=True)
```

</div>

<div class="sqlalchemy">

```python
session.add(Country(name='Spain', language='Spanish'))
```
</div>


### Update rows

<div class="preql">

```javascript
tips[tip < 2] update {tip: tip*2}
```

Preql puts the `update` keyword after the selection, so that when working interactively, you can first see which rows you're about to update.
</div>

<div class="sql">

```SQL
UPDATE tips SET tip = tip*2 WHERE tip < 2;
```

</div>

<div class="pandas">

```python
tips.loc[tips['tip'] < 2, 'tip'] *= 2
```
(takes a different form for complex operations)
</div>

## Gotchas

### Null checks

<div class="preql">

Comparisons to `null` behave like in Python.

```javascript
tips[col2==null]
```

Preql also has a value called `unknown`, which behaves like SQL's `NULL`.
</div>

<div class="sql">

Simple comparison to `NULL` using `=`, will always return `NULL`. For comparing to `NULL`, you must use the `IS` operator (the operator name changes between dialects).

```SQL
SELECT * FROM tips WHERE col2 IS NULL;
```
</div>


<div class="pandas">

```python
tips[tips['col2'].isna()]
```
</div>

## Programming

### Defining a function, and calling it from the query

<div class="preql">

```javascript
func add_one(x: int) = x + 1

my_table{ add_one(my_column) }
```

(Type annotations validate the values at compile-time)

</div>

<div class="sql">

(Postgres dialect)

```SQL
CREATE FUNCTION add_one(x int)
RETURNS int
AS
$$
 SELECT x + 1
$$
LANGUAGE SQL IMMUTABLE STRICT;

SELECT add_one(my_column) FROM my_table;
```
</div>

<div class="pandas">

```python
def add_one(x: int):
    return x + 1

my_table['my_column'].apply(add_one)
```
</div>

<div class="sqlalchemy">

Impossible?

</div>

### Counting a table from Python

This example demonstrates Preql's Python API.

All examples set `row_count` to an integer value.


<div class="preql">

(assumes `p` is a preql instance)

```javascript
row_count = len(p.my_table)
```

Or:

```javascript
row_count = p('count(my_table)')
```
</div>


<div class="sql">

```SQL
cur = conn.execute('SELECT COUNT() FROM my_table')
row_count = cur.fetchall()[0][0]
```
</div>

<div class="pandas">

```python
row_count = len(my_table.index)
```

<div class="sqlalchemy">

```python
row_count = session.query(my_table).count()
```

</div>