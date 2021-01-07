# Code comparison between Preql, Pandas, and SQL (the basics)

Based on the official [Pandas comparison with SQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)

## Selecting fields

In SQL, selection is done using the SELECT statement

```SQL
-- SQL
SELECT total_bill, tip, smoker, time FROM tips LIMIT 5;
```

With pandas, column selection is done by passing a list of column names to your DataFrame:

```python
# Python + Pandas
tips[['total_bill', 'tip', 'smoker', 'time']].head(5)
```

In Preql, column selection is done using the projection operator, `{}`:
```javascript
// Preql
tips{total_bill, tip, smoker, time}[..5]
```

Because the table name (`tips`) comes first, Preql can automatically suggest the field names.

## Filtering

Filtering in SQL is done via a WHERE clause.

```SQL
-- SQL
SELECT * FROM tips WHERE size >= 5 OR total_bill > 45;
```

DataFrames can be filtered in multiple ways; Pandas suggest using boolean indexing:

```python
# Python + Pandas
tips[(tips['size'] >= 5) | (tips['total_bill'] > 45)]
```

In Preql, we use the filter operator, `[]`:

```javascript
// Preql
tips[size >= 5 or total_bill > 45]
```

## Null checks

In SQL, simple comparison to `NULL` using `=`, will always return `NULL`. For comparing to `NULL`, you must use the `IS` operator.

```SQL
-- SQL
SELECT * FROM frame WHERE col2 IS NULL;
```

In Pandas, `NULL` checking is done using the `notna()` and `isna()` methods.

```python
# Python + Pandas
frame[frame['col2'].isna()]
```

In Preql, you simply compare to `null`, like you would in Python:

```javascript
// Preql
frame[col2 == null]
```

Preql also has a value called `unknown`, which behaves like SQL's `NULL`.

## Group by

Say we’d like to see how the amount of tips differs by day of the week. Here's how it might looks in SQL:

```SQL
-- SQL
SELECT day, AVG(tip), COUNT(*) FROM tips GROUP BY day;
```

The pandas equivalent would be:
```python
# Python + Pandas
tips.groupby('day').agg({'tip': np.mean, 'day': np.size})
```

Preql extends the projection operator to allow aggregation using the `=>` construct:

```javascript
// Preql
tips{day => avg(tip), count()}
```

Conceptually, everything on the left of `=>` are the keys, and on the right are the aggregated values.

## Join

Join is an operation that matches rows between two tables, based on common attributes.

Here's how an inner join might look in SQL:
```SQL
-- SQL
SELECT * FROM table1 INNER JOIN table2 ON table1.key = table2.key;
```

Here's the equivalent in Pandas: (it gets complicated if the key isn't with the same name)
```python
# Python + Pandas
pd.merge(df1, df2, on='key')
```

In Preql, you simply use the `join()` function:
```javascript
// Preql
join(a: table1.key, b: table2.key)
```

(The result is a table with two columns, `a` and `b`, which are structures (dicts) that each contain the columns of their respective table)

If we have pre-defined a "default join" between tables, we can just do:

```javascript
// Preql
join(a: table1, b: table2)
```


## Union and concat

Concat in SQL:

```SQL
-- SQL
SELECT * FROM df1 UNION ALL SELECT * FROM df2;
```

Concat in pandas:

```python
# Python + Pandas
pd.concat([df1, df2])
```

Concat in Preql:
```javascript
// Preql
df1 + df2
```

Union in SQL:
```SQL
-- SQL
SELECT * FROM df1 UNION SELECT * FROM df2;
```

Union in Pandas, as recommended on their website:

```python
# Python + Pandas
pd.concat([df1, df2]).drop_duplicates()
```

Union in Preql:
```javascript
// Preql
df1 | df2
```

## Top n rows with offset

SQL:

```SQL
-- SQL
SELECT * FROM tips ORDER BY tip DESC LIMIT 10 OFFSET 5;
```

Pandas:
```python
# Python + Pandas
tips.nlargest(10 + 5, columns='tip').tail(10)
```

Preql:
```javascript
// Preql
tips order {^tip} [5..15]
```

## Update rows

SQL:

```SQL
-- SQL
UPDATE tips SET tip = tip*2 WHERE tip < 2;
```

Pandas (takes a different form for complex operations):

```python
# Python + Pandas
tips.loc[tips['tip'] < 2, 'tip'] *= 2
```

Preql:

```javascript
// Preql
tips[tip < 2] update {tip: tip*2}
```

Preql puts the `update` keyword after the selection, so that when working interactively, you can first see which rows you're about to update.