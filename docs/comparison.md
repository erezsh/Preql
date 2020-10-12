# Comparison between Preql, SQL, and Pandas

// TODO show table results of Preql

Based on the official [Pandas comparison with SQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)

## Selecting fields

In SQL, selection is done using the SELECT statement

```SQL
SELECT total_bill, tip, smoker, time FROM tips LIMIT 5;
```

With pandas, column selection is done by passing a list of column names to your DataFrame:

```python
tips[['total_bill', 'tip', 'smoker', 'time']].head(5)
```

In Preql, column selection is done using the projection operator:
```javascript
tips{total_bill, tip, smoker, time}[..5]
```

Because the table name (`tips`) comes first, Preql can automatically suggest the field names.

## Filtering

Filtering in SQL is done via a WHERE clause.

```SQL
SELECT * FROM tips WHERE size >= 5 OR total_bill > 45;
```

DataFrames can be filtered in multiple ways; Pandas suggest using boolean indexing:

```python
tips[(tips['size'] >= 5) | (tips['total_bill'] > 45)]
```

In Preql, we use the filter operator:

```javascript
tips[size >= 5 or total_bill > 45]
```

## Null checks

In SQL, simple comparison to `NULL` using `=`, will always return `NULL`. For comparing to `NULL`, you must use the `IS` operator.

```SQL
SELECT * FROM frame WHERE col2 IS NULL;
```

In Pandas, `NULL` checking is done using the `notna()` and `isna()` methods.

```python
frame[frame['col2'].isna()]
```

In Preql, you simply compare to `null`, like you would in Python:

```javascript
frame[col2 == null]
```

Preql also has a value called `unknown`, which behaves like SQL's `NULL`.

## Group by

Say we’d like to see how tip amount differs by day of the week. Here's how it looks in SQL:

```SQL
SELECT day, AVG(tip), COUNT(*) FROM tips GROUP BY day;
```

The pandas equivalent would be:
```python
tips.groupby('day').agg({'tip': np.mean, 'day': np.size})
```

Preql extends the projection operator to allow aggregation using the `=>` construct:

```javascript
tips{day => avg(tip), count()}
```

Conceptually, everything on the left of `=>` are the keys, and on the right are the aggregated values.

## Join

Join is an operation that matches rows between two tables, based on common attributes.

Here's how an inner join might look in SQL:
```SQL
SELECT * FROM df1 INNER JOIN df2 ON df1.key = df2.key;
```

Here's the equivalent in Pandas: (it gets complicated if the key isn't with the same name)
```python
pd.merge(df1, df2, on='key')
```

In Preql, you simply use the `join()` function:
```javascript
join(a: df1.key, b: df2.key)
```

Unlike SQL and Pandas, which inline the rows together, Preql puts them side by side, in two structured columns. That's why we need to name the new columns (here `a` and `b`)

## Union and concat

Concat in SQL:

```SQL
SELECT * FROM df1 UNION ALL SELECT * FROM df2;
```

Concat in pandas:

```python
pd.concat([df1, df2])
```

Concat in Preql:
```javascript
df1 + df2
```

Union in SQL
```SQL
SELECT * FROM df1 UNION SELECT * FROM df2;
```

Union in Pandas, as recommended on their website:

```python
pd.concat([df1, df2]).drop_duplicates()
```

Union in Preql:
```javascript
df1 | df2
```

## Top n rows with offset

SQL:

```SQL
SELECT * FROM tips ORDER BY tip DESC LIMIT 10 OFFSET 5;
```

Pandas:
```python
tips.nlargest(10 + 5, columns='tip').tail(10)
```

Preql:
```javascript
tips order {^tip} [5..15]
```

## Top n rows per group

// TODO

## Update rows

SQL:

```SQL
UPDATE tips SET tip = tip*2 WHERE tip < 2;
```

Pandas:

```python
tips.loc[tips['tip'] < 2, 'tip'] *= 2
```

Preql:

```javascript
tips[tip < 2] update {tip: tip*2}
```

Preql puts the `update` keyword after the selection, so when working interactively, you can first see what rows you're going to update.