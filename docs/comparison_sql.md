# Code comparison between Preql and SQL

## Comparing to null

In Preql, `null` is treated as a value, instead of an unknown (a bit like `null` vs `undefined` in Javascript)

```javascript
count(Person[name==null])
```

```SQL
-- postgresql
SELECT COUNT(*) FROM Person WHERE name IS NOT DISTINCT FROM NULL
```


## Create lists on the fly

In Preql, you can define lists just like in Python or Javascript:

```go
lucky_numbers {n in [13, 27, 42]}
```

The closest equivalent in SQL:

```SQL
WITH my_list(item) AS (VALUES(13),(27),(42))
SELECT n in my_list FROM lucky_numbers
```

## Range

For when you want to create a sequence of numbers.

Preql:

```go
[1..10]
```

SQL:

```SQL
WITH RECURSIVE range AS (SELECT 1 AS item UNION ALL SELECT item+1 FROM range WHERE item+1<10)
SELECT * FROM range
```

## Better GROUP BY syntax

Preql lets you express aggragation as {key => value}, instead of SQL's clunky syntax.

```go
Person { country => name }
```

```SQL
-- postgreql
SELECT country, array_agg(name) AS name FROM Person GROUP BY country
```

## Using functions to factor repetitive code

In Preql, you can use functions to wrap repetitive code.

Here's some SQL code I found on github:

```SQL
-- sqlite
SELECT
    count(case when strftime('%w',author_when)='0' then 1 end) as sunday,
    count(case when strftime('%w',author_when)='1' then 1 end) as monday,
    count(case when strftime('%w',author_when)='2' then 1 end) as tuesday,
    count(case when strftime('%w',author_when)='3' then 1 end) as wednesday,
    count(case when strftime('%w',author_when)='4' then 1 end) as thursday,
    count(case when strftime('%w',author_when)='5' then 1 end) as friday,
    count(case when strftime('%w',author_when)='6' then 1 end) as saturday,
    author_email
FROM commits GROUP BY author_email
```

In Preql, it could be written as:

```go
func count_day(date, day) = SQL(int, "count(case when strftime('%w',$date)='$day' then 1 end)")

commits {
    author_email
    =>
    sunday: count_day(author_when, 0)
    monday: count_day(author_when, 1)
    tuesday: count_day(author_when, 2)
    wednesday: count_day(author_when, 3)
    thursday: count_day(author_when, 4)
    friday: count_day(author_when, 5)
    saturday: count_day(author_when, 6)
}
```
