![alt text](logo_small.png "Logo")

Preql (*pronounced: Prequel*) is an interpreted relational query language.

It is designed for use by data engineers, analyists and data scientists.

* Preql compiles to SQL at runtime. It has the performance and abilities of SQL, and much more.

* Programmer-friendly syntax and semantics, with gradual type-checking, inspired by Typescript and Python

* Interface through Python, HTTP or a terminal environment with autocompletion

* Escape hatch to SQL, for all those databse-specific features we didn't think to include

* Support for Postgres, MySQL and Sqlite. (more planned!)

**Note: Preql is still work in progress, and isn't ready for production use, or any serious use yet**

# Documentation

[Read here](https://preql.readthedocs.io/en/latest/)

# Get started

Simply install via pip:

```sh
    pip install -U prql
```

Then just run the interpeter:

```sh
    preql
```

Requires Python 3.8+

# Quick Example

```javascript
// Sum up all the squares of an aggregated list
 >> func sqrsum(x) = sum(x*x)
 >> [1..100]{ => sqrsum(value)}
table  =1
┌────────┐
│ sqrsum │
├────────┤
│ 328350 │
└────────┘
```

In the background, this was run by executing the following SQL code (reformatted):

```sql
WITH range1         AS (SELECT 1 AS value UNION ALL SELECT value+1 FROM range1 WHERE value+1<100)
   , subq_3(sqrsum) AS (SELECT SUM(value * value) AS sqrsum FROM range1)
SELECT * FROM subq_3
```


# Contributions

Code contributions are welcome!

By submitting a contribution, you assign to Preql all right, title and interest in any copyright you have in the Contribution, and you waive any rights, including any moral rights, that may affect our ownership of the copyright in the Contribution.

# License

Preql uses an “Interface-Protection Clause” on top of the MIT license.

See: [LICENSE](LICENSE)

In simple words, it can be used for any commercial or non-commercial purpose, as long as your product doesn't base its value on exposing the Preql language itself to your users.

If you want to add the Preql language interface as a user-facing part of your commercial product, contact us for a commercial license.
