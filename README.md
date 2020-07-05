![alt text](logo_small.png "Logo")

# A relational query language for engineers and scientists

Preql (pronounced: Prequel) is an interpreted relational query language.

It takes the best parts of SQL, Pandas and GraphQL, and combines them into a single Typescript-like language that compiles into SQL at run-time.

* Has the performance and abilities of SQL, and much more (* performance still needs work)
* Programmer friendly syntax and semantics, inspired by Python and Javascript
* Human-friendly errors, with gradual type-checking
* Interface through Python, HTTP or a REPL environment with autocompletion

# Alternative to SQL

* Client-side first-order functions
* Better errors
* Saner syntax and type-system

SQL() escape-hatch

# Alterative to an ORM

* Express yourself without syntax hacks and weird limitations
* Use the full power and speed of SQL
* Get your data already structured as a json (with nesting)

# Alternative to GraphQL, when it's applied a single SQL database

* More expressive

# Limitations

Limited by capabilities of database engine, performance & features

SQL() escape-hatch allows smart utilization of plugins


# Plans

* Mongo backend?
* Support for multiple concurrent backends?
* Clever query optimizations
* JSON support


-------------

Performance penalty is constant for each query

You always want your data to be as annotated as possible


# Notes

* Reqruires Python 3.8+

* Preql, PreQL, or PQL?