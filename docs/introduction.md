# Introduction

## Preface

Relational databases are a common and powerful approach to storing and processing information. Based on the solid foundation of relational algebra, they are efficient, resilient, well-tested, and full of useful features.

However, they all share the same weakness: They all use an antiquated programming interface called SQL.

While SQL was clever and innovative at the time of its conception, today we can look back on its design and see it has many fundamental mistakes, which make SQL incompatible with our contemporary notions of how a programming language should look and work.

As data becomes ever more important to the world of computation, so grows the appeal of a better solution. This need for an alternative inspired us to create the Preql programming language.

## Preql

Preql is a new programming language that aims to replace SQL as the standard language for programming databases.

These are big shoes to fill. Here is how Preql intends to do it:

### Preql compiles to SQL

Like SQL, Preql is guided by relational algebra, and is designed around operations on tables.

In Preql, `table` is a built-in type, and all table operations, such as filtering, sorting, and group-by, are compiled to SQL.

That means Preql code can run as fast and be as expressive as SQL.

Preql supports multiple targets, including Postgres, MySQL and Sqlite. See [features](features.md) for a complete list.

### Preql is interpreted

Not everything can be done in SQL. Control-flow constructs like for-loops, or downloading a remote JSON file, aren't possible in every database implementation.

Whenever your Preql code can't be compiled to SQL, it will be interpreted instead.

Being interpreted also lets Preql adopt advanced concepts, like "everything is an object", support for `eval()`, and so on.

That means in Preql you can do anything you could do in Python or Javascript, even when SQL can't.

### Better syntax, semantics, and practices

Preql's syntax is inspired by javascript. It's concise and natural for programmers.

It integrates important ideas like Fail-Early, and the Principle Of Least Astonishment.

Preql code is stored in files, instead of the database, which means it can be version-controlled (using git or similar)

It also comes with an interactive prompt, auto-complete, and IDE support (**TODO**)

### Escape hatch to SQL

There are many dialects of SQL, and even more plugins and extensions. Luckily, we don't need to support all of them.

Preql provides the builtin function `SQL()`, which allows you to run arbitrary SQL code.

The `SQL()` function can be embedded anywhere within your Preql code.

Here's a small example demonstrating this:

```javascript
>> func my_upper(s) = SQL(string, "upper($s) || '!'")
>> foo_bar = ["foo", "bar"]   // define a list
>> foo_bar{my_upper(item)}   // apply 'my_upper' to each item
["FOO!", "BAR!"]
```


## Preql vs *

### vs SQL

SQL first appeared in 1974, and aimed to provide a database interface that was based on natural language. It was clever and innovative at the time of its conception, but today we can look back on its design and see many fundamental mistakes.

Among its many faults, SQL is excessively verbose, is bad at catching and reporting errors, has no first-class or high-order functions, is awkward for interactive work, and it has a fragmented ecosystem and many different and incompatible dialects.

Some attempts have been made to address these issues, mainly in the form of ORMs.

### vs ORMs

ORMs are frameworks written within other programming languages, that let their users use sane and relatively concise phrases that in turn are compiled to SQL, and executed on the database.

However, they are themselves limited by their host languages, which were never designed for relational data processing. For the most part, they have awkward syntax, and they only support simple constructs and queries.

### vs Pandas

Given the failings of SQL and ORMs, it's no wonder that many programmers and data analysts choose to disregard relational database altogether, and use completely new approaches.

Pandas is one of those new approaches. Implemented entirely on top of Python and Numpy, it has gained a lot of popularity in recent years due to its accessibility and relative simplicity. It also has a wide range of features that were designed specifically for data scientists.

Unfortunately, it comes with its own set of faults. Pandas is slow (despite recent efforts to accelerate it), it has awkward syntax, it isn't well suited for relational, structured or linked data,


## Conclusion

Preql's design choices allow it to be fast and flexible, with an elegant and concise syntax.

While still young, it has the potential to one day replace SQL as the standard query language for relational databases.

To learn more about Preql, read the [tutorial](tutorial.md), or check out the [code comparison](comparison.md).
