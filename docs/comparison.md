# Code comparison: Preql, SQL and Pandas

This document was written with the aid of [Pandas' comparison with SQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html).

Use the checkboxes to hide/show the code examples for each language.

<script src="https://cdnjs.cloudflare.com/ajax/libs/cash/8.1.0/cash.min.js"></script>

<form>
<table>
	<tr>
		<td> <input type="checkbox" id="preql" onclick="$('.preql').toggle()" checked="checked"> </td>
		<td> <label for="preql">Preql</label> </td>
	</tr>
	<tr>
		<td> <input type="checkbox" id="sql" onclick="$('.sql').toggle()" checked="checked"> </td>
		<td> <label for="sql">SQL</label> </td>
	</tr>
	<tr>
		<td> <input type="checkbox" id="pandas" onclick="$('.pandas').toggle()" checked="checked"> </td>
		<td> <label for="pandas">Pandas</label> </td>
	</tr>
</table>
</form>

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
	.preql::before, .sql::before, .pandas::before {
	  font-weight: bold;
	  font-size: 1.1em;
	  color: #489;
	  margin-bottom: 5px;
	  text-decoration: underline;
	  line-height: 2em;
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

### Concat, Union

In this example, we will concatenate and union two tables together.

<div class="preql">

```javascript
df1 + df2	// concat
```

```javascript
df1 | df2	// union
```

</div>

<div class="sql">

```SQL
SELECT * FROM df1 UNION ALL SELECT * FROM df2;  -- concat
```

Union:
```SQL
SELECT * FROM df1 UNION SELECT * FROM df2;      -- union
```
</div>

<div class="pandas">

```python
pd.concat([df1, df2])                      # concat
```

```python
pd.concat([df1, df2]).drop_duplicates()    # union
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
(takes a different form for complex operations):
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

### Embedding functions in queries