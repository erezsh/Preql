![alt text](logo_small.png "Logo")

Preql (*pronounced: Prequel*) is an interpreted, relational programming language, that specializes in database queries.

It is designed for use by data engineers, analysts and data scientists.

Preql's main objective is to provide an alternative to SQL, in the form of a high-level programming language, with first-class functions, modules, strict typing, and Python integration.

**How does it work?**

Preql code is interpreted and gets compiled to SQL at runtime. This way, Preql gains the performance and abilities of SQL, but can also operate as a normal scripting language.

Currently supported dialects are:
* Postgres
* MySQL
* Sqlite
* BigQuery (soon)
* More... (planned)

For features that are database-specific, or aren't implemented in Preql, there is a `SQL()` function that provides a convenient escape hatch to write raw SQL code.

**Main Features**

* Modern syntax and semantics
    - Interpreted, everything is an object
    - Strong type system with gradual type validation and duck-typing
* Compiles to SQL
* Python and Pandas integration
* Interactive shell (REPL) with auto-completion
* Runs on Jupyter Notebook


**Note: Preql is still work in progress, and isn't ready for production use, or any serious use quite yet.**

# Learn More

- [**Read the documentation**](https://preql.readthedocs.io/en/latest/)

- [Follow the tutorial](https://preql.readthedocs.io/en/latest/tutorial.html)

- [Browse the examples](https://github.com/erezsh/Preql/tree/master/examples)


# Get started

Simply install via pip:

```sh
    pip install -U preql-lang
```

Then just run the interpreter:

```sh
    preql
```

Requires Python 3.8+

[Read more](https://preql.readthedocs.io/en/latest/getting-started.html)

# Quick Example

```javascript
// The following code sums up all the squares of an aggregated list of
// numbers, grouped by whether they are odd or even.

func sum_of_squares(x) = sum(x * x)
func is_even(x) = (x % 2 == 0)

// Create a list of [1, 2, 3, ..., 99]
num_list = [1..100]

// Group by is_even(), and run sum_of_squares() on the grouped values.
print num_list{ is_even(item) => sum_of_squares(item) }

// Result is:
┏━━━━━━━━━┳━━━━━━━━┓
┃ is_even ┃ sqrsum ┃
┡━━━━━━━━━╇━━━━━━━━┩
│       0 │ 166650 │
│       1 │ 161700 │
└─────────┴────────┘
```

In the background, this was run by executing the following compiled SQL code (reformatted):

```sql
  WITH range1 AS (SELECT 1 AS item UNION ALL SELECT item+1 FROM range1 WHERE item+1<100)
  SELECT ((item % 2) = 0) AS is_even, SUM(item * item) AS sqrsum FROM range1 GROUP BY 1;
```

# License

Preql uses an “Interface-Protection Clause” on top of the MIT license.

See: [LICENSE](LICENSE)

In simple words, it can be used for any commercial or non-commercial purpose, as long as your product doesn't base its value on exposing the Preql language itself to your users.

If you want to add the Preql language interface as a user-facing part of your commercial product, contact us for a commercial license.
