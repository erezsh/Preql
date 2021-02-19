# Why not SQL/ORM/Pandas ?

## SQL

SQL first appeared in 1974, and aimed to provide a database interface that was based on natural language. It was clever and innovative at the time of its conception, but today we can look back on its design and see many fundamental mistakes.

Among its many faults, SQL is excessively verbose, is bad at catching and reporting errors, has no first-class or high-order functions, is awkward for interactive work, and it has a fragmented ecosystem and many different and incompatible dialects.

Some attempts have been made to address these issues, mainly in the form of ORMs.

### Good parts of SQL

- Relational Algebra

- Declarative queries

- Many mature, well-tested implementations

- Intended for interactive work

### Bad parts of SQL

- Lack of first-class functions

- Hard to re-use code

- Bad error-handling (if any)

- Long-winded and clumsy syntax

- Code lives on the server, so there is no version control (such as git)

- Interactive clients leave a lot to be desired

(there are plenty more on both sides of the scales)

Preql adopts the good parts of SQL, and tries to solve the bad parts.


## ORMs

ORMs (object-relational mapping), are frameworks that let their users interact with the database using constructs that are native to the host programming language. Those constructs are then compiled to SQL, and executed in the database.

ORMs are usually more concise and more composable than SQL. However, they are themselves limited by their host languages, which were never designed for relational data processing. For the most part, they have awkward syntax, and they only support simple constructs and queries, and simplistic compostion.

## Pandas

Given the failings of SQL and ORMs, it's no wonder that many programmers and data analysts choose to disregard relational databases altogether, and use completely new approaches.

Pandas is one of those new approaches. Implemented entirely on top of Python and Numpy, it has gained a lot of popularity in recent years due to its accessibility and relative simplicity. It also has a wide range of features that were designed specifically for data scientists.

Unfortunately, it comes with its own set of faults. Pandas is slow (despite recent efforts to accelerate it), it has awkward syntax, and it isn't well suited for working with relational, structured or linked data.

See also: [Code comparison](comparison.html)

