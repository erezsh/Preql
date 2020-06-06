#  Tutorial for the basics of Preql

## What is Preql?

Preql (pronounced: Prequel) is a new programming language for relational databases.

It lets you use syntax and semantics that are similar to Javascript and Python, but compiles to SQL queries, and runs them on a database backend.

By combining these elements, Preql lets you write simple and elegant code that runs as fast as SQL.

We'll soon dive into the language itself, let's first let's install and learn how to use the interpreter

## Getting Started (Install & How to use)

You can install preql by running this in your shell/command prompt:

```sh
$ pip install preql
```

Usually, you would connect Preql to a database, or load an existing module.

But, you can also access the preql interpreter like this:

```sh
$ python -m preql
Welcome to the Preql REPL. Type help() for help
>>
```

By default, the interpreter uses SQLite's memory database. We'll later see how to change it using the `connect()` function.

From now on, we'll use `>>` to signify the Preql REPL

Press Ctrl+C to interrupt an existing operation or prompt. Press Ctrl+D or run `exit()` to exit the interpreter.

You can run the `names()` function to see what functions are available.

You can also run a preql file. Let's edit a file called `helloworld.pql`:

```go
// helloworld.pql
print "Hello World!"
```

And then run it:

```go
$ python -m preql helloworld.pql
Hello World!
```

## Basic Expressions

Preql has integers, floats and strings. They behave as you would expect:

```go
>> 1 + 1
2
>> 2 / 4
0.5
>> 27 % 13
1
>> "a" + "b"
"ab"
>> "-" * 5
"-----"

>> (not 0) and 2 < 4
True
>> null == null     // Unlike SQL!
True
```

Notice that dividing two integers results in a float. To get an integer, use the `/~` operator:

```go
>> 10 /~ 3
3
```

You can get the type of anything in Preql by using the `type()` function:

```go
>> type(10)
int
>> type(int)
type
```

## Functions

Declare functions using func:

```go
func sign(x) {
  if (x == 0) {
    return 0
  }
  if (x > 0) {
    return 1
  }
  return -1
}

>> sign(100)
1
```

There's also a shorthand for "one-liners":

```go
>> func str_concat(s1, s2) = s1 + s2
>> str_concat("foo", "bar")
"foobar"
>> str_concat     // Functions are objects just like everything else
<func str_concat(s1, s2) = ...>
```

## Tables

Tables are basically a set of columns, that can be instanciated into a list of rows.

Preql's tables are stored in an SQL database, and most operations on them are done using SQL queries.


Here is how we would define a table of points:

```go
table Point {
    x: float
    y: float
}
```

This statement creates a permanent table in your database (if you are connected to one. The default database resides in memory and isn't persistent)

For this tutorial, let's create a table that's little more meaningful, and populate it with values:

```go
table Country {
  name: string
  population: int
}

palau = new Country("Palau", 17900)
nauru = new Country("Nauru", 11000)
new Country("Tuvalu", 10200)
```

We assigned the newly created rows to variables, but they also exist independently.

We can see that the `Country` table has three rows:

```go
>> count(Country)
3
```

The `new` statements inserted our values into an SQL table, and `count()` ran the following query: `SELECT COUNT(*) FROM Country`

(Note: You can see every SQL statement that's executed by starting the REPL with the `-d` switch.)

We can also observe the variables, or the entire table:

```go
 >> palau
Row{id: 1, name: "Palau", population: 17900}
>> palau.population + 1
17901
>> Country
table Country, count=3
  id  name      population
----  ------  ------------
   1  Palau          17900
   2  Nauru          11000
   3  Tuvalu         10200
```

Notice that every table automatically gets an `id` column. It's a useful practice, that provides us with an easy and performant "pointer" to refer to rows.

### Table operations

There are many operations that you can perform on a table. Here we'll go through the main ones.

**Selection** lets us filter tables using the selection operator:

```go
// All countries that contain the letter 'l' and a population below 15000
>> Country[name ~ "%l%", population < 15000]
table Country, count=1
  id  name      population
----  ------  ------------
   3  Tuvalu         10200
```

We can also filter the rows by index (zero-based), by providing it with a `range` instead.

```go
>> Country[1..]
table Country, count=2
  id  name      population
----  ------  ------------
   2  Nauru          11000
   3  Tuvalu         10200
```

Notice that the row index and the value of the `id` column are not related in any meaningful way.

We can also use functions inside table expressions, as long as they don't change the global state.

```javascript
>> func startswith(s, p) = s ~ (p + "%")    // Pattern match the string (LIKE)
>> my_list = ["cat", "dog", "car"]          // Define a list
>> new_list = my_list[startswith(value, "c")]  // Apply selection. `value` refers to the list's elements
>> print new_list                           // Execute SQL query
["cat", "car"]
```

Lists are basically tables with a single column named `value`.

**Projection** lets us create new tables, with columns of our own choice:

```go
>> Country{name, is_big: population>15000}
table Country_proj3, count=3
name      is_big
------  --------
Palau          1
Nauru          0
Tuvalu         0

>> func half(n) = n / 2
>> Country{..., half(population)}   // Ellipsis fills in all columns
table Country_proj58, count=3
  id  name      population    half
----  ------  ------------  ------
   1  Palau          17900    8950
   2  Nauru          11001  5500.5
   3  Tuvalu         10201  5100.5
```

Notice that Preql creates a new table type for each projection.

**Aggregation** looks a lot like projection, and lets us aggregate information:

The syntax is basically `{ keys => values }`

```go
// Count how many countries there are, for each length of name.
>> Country { length(name) => count(id) }
table Country_proj19, count=2
  length    count
--------  -------
       5        2
       6        1

// If no keys are given, aggregate all the rows into one.
>> world_population = Country { => sum(population) }
table Country_proj11, count=1
  sum
-----
39100

// Create an even-odd histogram
>> [1,2,3,4,5,6,7] {
      odd: value % 2 => count(value)
   }
table list_int_proj37, count=2
  odd    count
-----  -------
    0        3
    1        4

// Sum up all the squares
>> func sqrsum(x) = sum(x*x)
>> [1,2,3,4]{ => sqrsum(value)}
30
```

**Ordering** lets us sort the rows into a new table.

```go
>> Country order {population}
table Country, count=3
  id  name      population
----  ------  ------------
   3  Tuvalu         10200
   2  Nauru          11000
   1  Palau          17900

>> Country order {^name}
table Country, count=3
  id  name      population
----  ------  ------------
   3  Tuvalu         10200
   1  Palau          17900
   2  Nauru          11000
```

### Lazy-evaluation vs Temporary tables

Immutable table operations, such as selection and projection, are lazily-evaluated in Preql. That means that they don't execute until strictly necessary.

This allows for gradual chaining, that the compiler will then know to merge into a single query:

```go
a = some_table[x > 100]   // No SQL executed
b = a {x => sum(y)}       // ... same here
first20 = b[..20]         // ... same here
print first20             // Executes a single SQL query for all previous statements
print first20             // Executes the same SQL query all over again.
```

Lazy-evaluation for queries has the following advantages:
* Results in better compilation
* Leaner memory use, since we don't store intermediate results
* The results of the query are 'live', and update whenever the source table updates.

However, in situations when the same query is used in several different statements, it may be inefficient to run the same query again and again.

In those situations it may be useful to store the results in a temporary table:

```go
table first20 = b[..20]   // Execute a single SQL query and store it
print first20             // Only needs to query the 'first20' table
print first20             // Only needs to query the 'first20' table

```

A temporary table is a table that's persistent in the database memory for as long as the session is alive.

Here's another example:

```go
// Create a temporary table that resides in database memory
>> table t_names = Country[population>100]{name}  // Evaluated here once
>> count(t_names) + count(t_names)

// Create a query through lazy-evaluation. It's just a local definition
>> q_names = Country[population>100]{name}
>> count(q_names) + count(q_names)                // Evaluated here twice
```

The main disadvantage of using temporary tables is that they may fill up the database memory when working with large tables.

### Update

We can **update** tables in-place.

Updates are evaluated immediately. This is true for all expressions that change the global state.

Example:
```go
>> Country update {population: population + 1}
table Country, count=3
  id  name      population
----  ------  ------------
   1  Palau          17901
   2  Nauru          11001
   3  Tuvalu         10201

 >> Country[name=="Palau"] update {population: population - 1}
table Country, count=1
  id  name      population
----  ------  ------------
   1  Palau          17900
 >> Country
table Country, count=3
  id  name      population
----  ------  ------------
   1  Palau          17900
   2  Nauru          11001
   3  Tuvalu         10201
```

### Join

Joining two tables means returning a new table that contains the rows of both tables, matched on a certain attribute.

It is possible to omit the attributes when there is a predefined relationship between the tables.

```go
>> table odds = [1, 3, 5, 7, 9, 11]
>> table primes = [2, 3, 5, 7, 11]

// Creates columns `o` and `p`, which are structures containing the original rows.
>> join(o: odds.value, p: primes.value)
table join9, count=4
o                       p
----------------------  ----------------------
{'value': 3, 'id': 2}   {'value': 3, 'id': 2}
{'value': 5, 'id': 3}   {'value': 5, 'id': 3}
{'value': 7, 'id': 4}   {'value': 7, 'id': 4}
{'value': 11, 'id': 6}  {'value': 11, 'id': 5}

// We can of course destructure it into a regular table
>> join(o: odds.value, p: primes.value) {o.value, o_id: o.id, p_id: p.id}
table join33_proj34, count=4
  value    o_id    p_id
-------  ------  ------
      3       2       2
      5       3       3
      7       4       4
     11       6       5

// We can filter countries by name, by joining on their name:
>> join(c: Country.name, n:["Palau", "Nauru"].value) {c.id, c.name}
table join30_proj31, count=2
  id  name
----  ------
   1  Palau
   2  Nauru

// But of course the best way to accomplish this is to use the `in` operator
>> Country[name in ["Palau", "Nauru"]]
table Country, count=2
  id  name      population
----  ------  ------------
   1  Palau          17900
   2  Nauru          11001

// Or not in
>> Country[name !in ["Palau", "Nauru"]]
table Country, count=1
  id  name      population
----  ------  ------------
   3  Tuvalu         10201
```

## The SQL Escape-hatch

Preql does not, and cannot, implement every SQL function and feature.

There are too many dialects of SQL, and too little Preql programmers (for now).

Luckily, there is an escape hatch, through the `SQL()` function.

The first argument is the type of the result, and the second argument is a string of SQL code.

```go
>> func do_sql_stuff(x) = SQL(string, "lower($x) || '!'")   // Runs in Sqlite
>> ["UP", "Up", "up"]{ do_sql_stuff(value) }
table list_string_proj70, count=3
do_sql_stuff
--------------
up!
up!
up!
```

We can also query entire tables:

```go
 >> SQL(Country, "SELECT * FROM $Country WHERE name == \"Palau\"")
table Country, count=1
  id  name      population
----  ------  ------------
   1  Palau          17900
```

Notice that "Country" is used twice in different contexts: once as the return type, and once for its rows.

In fact, many of Preql's core functions are written using the `SQL()` function, for example `enum`:

```go
func enum(tbl) {
    "Return the table with a new index column"
    // Uses SQL's window functions to calculate the index per each row
    // Remember, ellipsis (...) means include all columns.
    return tbl{index: SQL(int, "row_number() over ()"), ...}
}

// Add index for each row
>> enum(Country order {population})
table Country_proj80, count=3
  index    id  name      population
-------  ----  ------  ------------
      1     3  Tuvalu         10201
      2     2  Nauru          11001
      3     1  Palau          17900
```

## Notable Built-in functions

Here is a partial list of functions provided by Preql:

- `debug()` - call this from your code to drop into the interpreter. Inside, you can use `c()` or `continue()` to resume running.
- `import_csv(table, filename)` - import the contents of a csv file into an existing table
- `random()` - return a random number
- `now()` - return a `datetime` object for now
- `sample_fast(tbl, n)` - return a sample of `n` rows from table `tbl` (O(n), maximum of two queries). May introduce a minor bias (See `help(sample_fast)`).
- `count_distinct(field)` - count how many unique values are in the given field/column.

## Calling Preql from Python

Preql is not only a standalone tool, but also a Python library. It can be used as an alternative to ORM libraries such as SQLAlchemy.

It's as easy as:

```python
>>> import preql
>>> p = preql.Preql()
```

You can also specify which database to work on, and other parameters.

Then, just start working by calling the object with Preql code:

```python
# Use the result like in an ORM
>>> len(p('[1,2,3][value>=2]'))
2

# Turn the result to JSON (lists and dicts)
>>> p('[1,2]{type: "example", values: {v1: value, v2: value*2}}').to_json()
[{'type': 'example', 'values': {'v1': 1, 'v2': 2}}, {'type': 'example', 'values': {'v1': 2, 'v2': 4}}]

# Run Preql code file
>>> p.load('some_file.pql')
```

You can also reach variables inside the Preql namespace using:

```python
>>> p('a = [1,2,3]')
>>> sum(p.a)
6
>>> p.names()[:10]  # `names()` is a global function, like Python's `dir()`
['int', 'float', 'string', 'text', 'bool', 'datetime', 'exit', 'help', 'names', 'dir']
```

### Using Pandas

You can easily import/export tables between Preql and Pandas, by using Python as a middleman:

```python
>>> from pandas import DataFrame
>>> import preql
>>> p = preql.Preql()
>>> f = DataFrame([[1,2,"a"], [4,5,"b"], [7,8,"c"]], columns=['x', 'y', 'z'])

>>> x = p.import_pandas(x=f)
>>> p.x                         # Returns a Preql table
table x, =3
  id    x    y  z
----  ---  ---  ---
   1    1    2  a
   2    4    5  b
   3    7    8  c

>>> p('x {y, z}').to_pandas()    # Returns a pandas table
   y  z
0  2  a
1  5  b
2  8  c

>>> p('x{... !id}').to_pandas() == f  # Same as it ever was
      x     y     z
0  True  True  True
1  True  True  True
2  True  True  True
```
