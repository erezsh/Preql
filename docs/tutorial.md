#  Tutorial for the basics of Preql

## What is Preql?

Preql is a relational programming language that runs on relational databases, such as PostgreSql, MySql, and Sqlite.

It does so by compiling to SQL. However, its syntax and semantics resemble those of Javascript and Python.

By combining these elements, Preql lets you write simple and elegant code that runs as fast as SQL.

We'll soon dive into the language itself, but first let's install and learn how to use the interpreter

## Getting Started (Install & How to use)

You can install preql by running this in your shell/command prompt:

```sh
$ pip install -U preql-lang
```

Usually, you would connect Preql to a database, or load an existing module.

But, you can also just run the preql interpreter as is:

```sh
$ preql
Preql 0.1.16 interactive prompt. Type help() for help
>>
```

By default, the interpreter uses SQLite's memory database. We'll later see how to change it using the `connect()` function.

From now on, we'll use `>>` to signify the Preql REPL.

Press Ctrl+C at any time to interrupt an existing operation or prompt. Press Ctrl+D or run `exit()` to exit the interpreter.

You can run the `names()` function to see what functions are available, and `help()` to get interactive help.

You can also run a preql file. Let's create a file called `helloworld.pql`:

```javascript
// helloworld.pql
print "Hello World!"
```

And then run it:

```javascript
$ preql -m helloworld
Hello World!
```

Alternatively, we could do `preql -f helloworld.pql`, if we want to specify a full path.

We can also use Preql as a Python library:

```python
# In the Python interpreter
from preql import Preql
p = Preql()

assert p('sum([1..10])') == 45

p('''
  func my_range(x) = [1..x]
''')
print(p.my_range(8))
# Output:
# [1, 2, 3, 4, 5, 6, 7]
```


## Basic Expressions

Preql has integers, floats and strings, which behave similarly to Python

`null` behaves just like Python's None.

```javascript
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
>> null + 1
Exception traceback:
  ~~~ At '<repl>' line 1, column 6
null + 1
-----^---

TypeError: Operator '+' not implemented for nulltype and int
```

Notice that dividing two integers results in a float. To get an integer, use the `/~` operator, which is equivalent to Python's `//` operator:

```javascript
>> 10 /~ 3
3
```

You can get the type of anything in Preql by using the `type()` function:

```javascript
>> type(10)
int
>> type(int)
type
```

Preql also has lists, which are essentially a table with a single column called `item`:

```javascript
 >> my_list = [1,2,3]
 table
   =3
┏━━━━━━┓
┃ item ┃
┡━━━━━━┩
│    1 │
│    2 │
│    3 │
└──────┘
 >> count(my_list + [4,5,6])
6
 >> names(my_list)
       table  =1
┏━━━━━━┳━━━━━━┳━━━━━┓
┃ name ┃ type ┃ doc ┃
┡━━━━━━╇━━━━━━╇━━━━━┩
│ item │ int  │     │
└──────┴──────┴─────┘
 >> type(my_list)
list[int]
 >> type(["a", "b", "c"])
list[string]
```

The range syntax creates a list of integers:

```javascript
>> [1..100]
table  =99
┏━━━━━━━━┓
┃   item ┃
┡━━━━━━━━┩
│      1 │
│      2 │
│      3 │
│      4 │
│      5 │
│    ... │
└────────┘
```

Preql only shows us a preview of the table. If we want to see more items, we can just enter a dot (`.`) in the prompt:

```javascript
>> .
table [5..] =99
┏━━━━━━━━┓
┃   item ┃
┡━━━━━━━━┩
│      6 │
│      7 │
│      8 │
│      9 │
│     10 │
│    ... │
└────────┘
```

Entering `.` again will keep scrolling more items.

### inspect_sql

You might be curious what SQL statements are being executed behind the scenes. You can find out using the `inspect_sql()`  function.

```javascript
 >> print inspect_sql([1..10] + [20..30])
```

```sql
WITH RECURSIVE range1 AS (SELECT 1 AS item UNION ALL SELECT item+1 FROM range1 WHERE item+1<10)
    , range2 AS (SELECT 20 AS item UNION ALL SELECT item+1 FROM range2 WHERE item+1<30)
    SELECT * FROM [range1] UNION ALL SELECT * FROM [range2] LIMIT -1
```

## Functions

Declare functions using func:

```javascript
func sign(x) {
  if (x == 0) {
    return 0
  } else if (x > 0) {
    return 1
  } else {
    return -1
  }
}
>> sign(-100)
-1
>> sign(100)
1
```

You can also use them in table operations!

```javascript
 >> [-20, 0, 30]{ sign(item) }
 table
   =3
┏━━━━━━┓
┃ sign ┃
┡━━━━━━┩
│   -1 │
│    0 │
│    1 │
└──────┘
```

Let's inspect the SQL code that is executed:

```javascript
 >> print inspect_sql([-20, 0, 30]{ sign(item) })
```

```SQL
WITH RECURSIVE list_1([item]) AS (VALUES (-20), (0), (30))
    SELECT CASE WHEN ([item] = 0) THEN 0 ELSE CASE WHEN ([item] > 0) THEN 1 ELSE -1 END  END  AS [sign] FROM [list_1]
```

Note: Functions with side-effects or I/O operations aren't allowed in table operations, due to SQL's limitations.

There's also a shorthand for "one-liners":

```javascript
>> func str_concat(s1, s2) = s1 + s2
>> str_concat("foo", "bar")
"foobar"
```

Functions are objects just like everything else, and can be passed around to other functions.

Here is a toy example that demonstrates this:

```javascript
func apply_function(f, x) = f(x)

my_list = ["this", "is", "a", "list"]

// Run `apply_function` for each item, and use the built-in `length` function for strings.
// `len` is just the name of the new column.
print my_list{
  len: apply_function(length, item)
}
// Output:
// table  =4
// ┏━━━━━━━┓
// ┃   len ┃
// ┡━━━━━━━┩
// │     4 │
// │     2 │
// │     1 │
// │     4 │
// └───────┘
```

## Tables

Tables are essentially a list of rows, where all the rows have the same structure.

That structure is defined by a set of columns, where each column has a name and a type.

Preql's tables are stored in an SQL database, and most operations on them are done efficiently using SQL queries.


Here is how we would define a table of points:

```javascript
table Point {
    x: float
    y: float
}
```

This statement creates a persistent table named `Point` in your database (if you are connected to one. The default database resides in memory and isn't persistent). The executed SQL looks like this:

```SQL
CREATE TABLE IF NOT EXISTS "Point" ("id" INTEGER, "x" FLOAT NOT NULL, "y" FLOAT NOT NULL, PRIMARY KEY (id))
```

If the table `Point` already exists, it will instead verify that the new definition is a subset of the existing one. That is, that all the columns defined in it exist in the current table, and with the correct type.

For this tutorial, let's create a table that's little more meaningful, and populate it with values:

```javascript
table Country {
  name: string
  population: int
}

palau = new Country("Palau", 17900)
nauru = new Country("Nauru", 11000)
new Country("Tuvalu", 10200)
```

`new` accepts its parameters in the order that they were defined in the table. However, it's also possible to use named arguments, such as `new Point(y:10, x:1)`.

In the above example, we assigned the newly created rows to variables. But they also exist independently in the table.

We can see that the `Country` table has three rows:

```javascript
>> count(Country)
3
```

The `new` statements inserted our values into an SQL table, and `count()` ran the following query: `SELECT COUNT(*) FROM Country`

(Note: You can see every SQL statement that's executed by starting the REPL with the `--print-sql` switch.)

We can also observe the variables, or the entire table:

```javascript
 >> palau
Row{id: 1, name: "Palau", population: 17900}
>> palau.population + 1
17901
>> Country
      table Country =3
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau  │      17900 │
│  2 │ Nauru  │      11000 │
│  3 │ Tuvalu │      10200 │
└────┴────────┴────────────┘
```

Notice that every table automatically gets an `id` column. It's a useful practice, that provides us with an easy and performant "pointer" to refer to rows.

### Table operations

There are many operations that you can perform on a table. Here we'll javascript through the main ones.

**Selection** lets us filter tables using the selection operator:

```javascript
// All countries that contain the letter 'l' and a population below 15000
>> Country[name like "%l%", population < 15000]
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  3 │ Tuvalu │      10200 │
└────┴────────┴────────────┘
```

We can chain table operations:

```javascript
 >> Country[name like "%l%" or population < 11000] {name, population}
       table  =2
┏━━━━━━━━┳━━━━━━━━━━━━┓
┃ name   ┃ population ┃
┡━━━━━━━━╇━━━━━━━━━━━━┩
│ Palau  │      17900 │
│ Tuvalu │      10200 │
└────────┴────────────┘
```

We can also filter the rows by index (zero-based), by providing it with a `range` instead.

```javascript
 >> Country[1..]
      table Country =2
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  2 │ Nauru  │      11000 │
│  3 │ Tuvalu │      10200 │
└────┴────────┴────────────┘
```

Notice that the row index and the value of the `id` column are not related in any meaningful way.

**Projection** lets us create new tables, with columns of our own choice:

```javascript
>> Country{name, is_big: population>15000}
     table  =3
┏━━━━━━━━┳━━━━━━━━┓
┃ name   ┃ is_big ┃
┡━━━━━━━━╇━━━━━━━━┩
│ Palau  │      1 │
│ Nauru  │      0 │
│ Tuvalu │      0 │
└────────┴────────┘
>> Country[name like "P%"]{name, is_big: population>15000}
    table  =1
┏━━━━━━━┳━━━━━━━━┓
┃ name  ┃ is_big ┃
┡━━━━━━━╇━━━━━━━━┩
│ Palau │      1 │
└───────┴────────┘
>> func half(n) = n / 2
>> Country{..., half(population)}   // Ellipsis fills in the rest of the columns
              table  =3
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ id ┃ name   ┃ population ┃   half ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│  1 │ Palau  │      17900 │ 8950.0 │
│  2 │ Nauru  │      11000 │ 5500.0 │
│  3 │ Tuvalu │      10200 │ 5100.0 │
└────┴────────┴────────────┴────────┘
```

Notice that Preql creates a new table type for each projection. Therefore, the fields that aren't included in the projection, won't be available afterwards.

However these are only types, and not actual tables. To create a persistent table, we can write:

```javascript
table half_population = Country{..., half(population)}
```

Now, if connected to a database, `half_population` will be stored persistently.

**Aggregation** looks a lot like projection, and lets us aggregate information:

The syntax is basically `{ keys => values }`

```javascript
// Count how many countries there are, for each length of name.
>> Country { length(name) => count(id) }
    table  =2
┏━━━━━━━━┳━━━━━━━┓
┃ length ┃ count ┃
┡━━━━━━━━╇━━━━━━━┩
│      5 │     2 │
│      6 │     1 │
└────────┴───────┘

// If no keys are given, aggregate all the rows into one.
>> world_population = Country { => sum(population) }
table  =1
┏━━━━━━━┓
┃   sum ┃
┡━━━━━━━┩
│ 39100 │
└───────┘

// We can extract the row from the table using the `one` operator
>> one world_population
Row{sum: 39100}

// Create an even-odd histogram
>> [1,2,3,4,5,6,7] {
      odd: item % 2 => count(item)
   }
   table  =2
┏━━━━━┳━━━━━━━┓
┃ odd ┃ count ┃
┡━━━━━╇━━━━━━━┩
│   0 │     3 │
│   1 │     4 │
└─────┴───────┘

// Sum up all the squares
>> func sqrsum(x) = sum(x*x)
>> [1,2,3,4]{ => sqrsum(item)}
table  =1
┏━━━━━━━━┓
┃ sqrsum ┃
┡━━━━━━━━┩
│     30 │
└────────┘
```

**Ordering** lets us sort the rows into a new table.

```javascript
>> Country order {population} // Sort ascending
      table Country =3
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  3 │ Tuvalu │      10200 │
│  2 │ Nauru  │      11000 │
│  1 │ Palau  │      17900 │
└────┴────────┴────────────┘

>> Country order {^name}      // Sort descending (^)
      table Country =3
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  3 │ Tuvalu │      10200 │
│  1 │ Palau  │      17900 │
│  2 │ Nauru  │      11000 │
└────┴────────┴────────────┘
```

### Lazy-evaluation vs Temporary tables

Immutable table operations, such as selection and projection, are lazily-evaluated in Preql. That means that they don't execute until strictly necessary.

This allows for gradual chaining, that the compiler will then know to merge into a single query:

```javascript
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

```javascript
table first20 = b[..20]   // Execute a single SQL query and store it
print first20             // Only needs to query the 'first20' table
print first20             // Only needs to query the 'first20' table

```

A temporary table is a table that's persistent in the database memory for as long as the session is alive.

Here's another example:

```javascript
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
```javascript
>> Country update {population: population + 1}
      table Country =3
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau  │      17901 │
│  2 │ Nauru  │      11001 │
│  3 │ Tuvalu │      10201 │
└────┴────────┴────────────┘

 >> Country[name=="Palau"] update {population: population - 1}
      table Country =1
┏━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name  ┃ population ┃
┡━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau │      17900 │
└────┴───────┴────────────┘
 >> Country
       table Country =3
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau  │      17900 │
│  2 │ Nauru  │      11001 │
│  3 │ Tuvalu │      10201 │
└────┴────────┴────────────┘
```

### Join

Joining two tables means returning a new table that contains the rows of both tables, matched on a certain attribute.

It is possible to omit the attributes when there is a predefined relationship between the tables.

```javascript
// Create tables from lists. That automatically adds an `id` column.
>> table odds = [1, 3, 5, 7, 9, 11]
>> table primes = [2, 3, 5, 7, 11]

// Join into columns `o` and `p`, which are structures containing the original rows.
>> join(o: odds.item, p: primes.item)
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ o                     ┃ p                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ {'item': 3, 'id': 2}  │ {'item': 3, 'id': 2}  │
│ {'item': 5, 'id': 3}  │ {'item': 5, 'id': 3}  │
│ {'item': 7, 'id': 4}  │ {'item': 7, 'id': 4}  │
│ {'item': 11, 'id': 6} │ {'item': 11, 'id': 5} │
└───────────────────────┴───────────────────────┘

// We can then destructure it into a regular table
>> join(o: odds.item, p: primes.item) {o.item, o_id: o.id, p_id: p.id}
      table  =4
┏━━━━━━┳━━━━━━┳━━━━━━┓
┃ item ┃ o_id ┃ p_id ┃
┡━━━━━━╇━━━━━━╇━━━━━━┩
│    3 │    2 │    2 │
│    5 │    3 │    3 │
│    7 │    4 │    4 │
│   11 │    6 │    5 │
└──────┴──────┴──────┘

// We can filter countries by name, by joining on their name:
>> join(c: Country.name, n:["Palau", "Nauru"].item) {...c}
         table  =2
┏━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name  ┃ population ┃
┡━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau │      17900 │
│  2 │ Nauru │      11001 │
└────┴───────┴────────────┘

// But idiomatically, the best way to accomplish this is to use the `in` operator
>> Country[name in ["Palau", "Nauru"]]
     table Country =2
┏━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name  ┃ population ┃
┡━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau │      17900 │
│  2 │ Nauru │      11001 │
└────┴───────┴────────────┘

// Or not in
>> Country[name !in ["Palau", "Nauru"]]
      table Country =1
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name   ┃ population ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│  3 │ Tuvalu │      10201 │
└────┴────────┴────────────┘
```

## The SQL Escape-hatch

Preql does not, and cannot, implement every SQL function and feature.

There are too many dialects of SQL, and too few Preql programmers (for now).

Luckily, there is an escape hatch, through the `SQL()` function.

The first argument is the type of the result, and the second argument is a string of SQL code.

```javascript
>> func do_sql_stuff(x) = SQL(string, "lower($x) || '!'")   // Runs in Sqlite
>> ["UP", "Up", "up"]{ do_sql_stuff(item) }
   table  =3
┏━━━━━━━━━━━━━━┓
┃ do_sql_stuff ┃
┡━━━━━━━━━━━━━━┩
│ up!          │
│ up!          │
│ up!          │
└──────────────┘
```

We can also query entire tables:

```javascript
 >> SQL(Country, "SELECT * FROM $Country WHERE name == \"Palau\"")
      table Country =1
┏━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ id ┃ name  ┃ population ┃
┡━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│  1 │ Palau │      17900 │
└────┴───────┴────────────┘
```

Notice that "Country" is used twice in different contexts: once as the return type, and once for querying its rows.

In fact, many of Preql's core functions are written using the `SQL()` function, for example `enum`:

```javascript
func enum(tbl) {
    "Return the table with a new index column"
    // Uses SQL's window functions to calculate the index per each row
    // Remember, ellipsis (...) includes all available columns.
    return tbl{
      index: SQL(int, "row_number() over ()")
      ...
    }
}

// Add an index for each row in the table
>> enum(Country order {population})
             table  =3
┏━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ index ┃ id ┃ name   ┃ population ┃
┡━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│     0 │  3 │ Tuvalu │      10201 │
│     1 │  2 │ Nauru  │      11001 │
│     2 │  1 │ Palau  │      17900 │
└───────┴────┴────────┴────────────┘
```

## Notable Built-in functions

Here is a partial list of functions provided by Preql:

- `debug()` - call this from your code to drop into the interpreter. Inside, you can use `c()` or `continue()` to resume running.
- `import_csv(table, filename)` - import the contents of a csv file into an existing table
- `random()` - return a random number
- `now()` - return a `datetime` object for now
- `sample_fast(tbl, n)` - return a sample of `n` rows from table `tbl` (O(n), maximum of two queries). May introduce a minor bias (See `help(sample_fast)`).
- `bfs(edges, initial)` - performs a breadth-first search on a graph using SQL
- `count_distinct(field)` - count how many unique values are in the given field/column.

To see the full list, run the following in Preql: `names(__builtins__)[type like "function%"]`

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
>>> len(p('[1,2,3][item>=2]'))
2

# Turn the result to JSON (lists and dicts)
>>> p('[1,2]{type: "example", values: {v1: item, v2: item*2}}').to_json()
[{'type': 'example', 'values': {'v1': 1, 'v2': 2}}, {'type': 'example', 'values': {'v1': 2, 'v2': 4}}]

# Run Preql code file
>>> p.load('some_file.pql')
```

You can also reach variables inside the Preql namespace using:

```python
>>> p('a = [1,2,3]')
>>> sum(p.a)
6
>>> p.char_range('w', 'z')    # char_range() is a built-in function in Preql
['w', 'x', 'y', 'z']
```

### Using Pandas

You can easily import/export tables between Preql and Pandas, by using Python as a middleman:

```python
>>> from pandas import DataFrame
>>> import preql
>>> p = preql.Preql()
>>> f = DataFrame([[1,2,"a"], [4,5,"b"], [7,8,"c"]], columns=['x', 'y', 'z'])

>>> x = p.import_pandas(x=f)
>>> p.x                          # Returns a Preql table
[{'x': 1, 'y': 2, 'z': 'a', 'id': 1}, {'x': 4, 'y': 5, 'z': 'b', 'id': 2}, {'x': 7, 'y': 8, 'z': 'c', 'id': 3}]

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



