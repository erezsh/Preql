![alt text](logo_small.png "Logo")

# Introducing PreQL

PreQL (pronounced: Prequel) is an interpreted relational query language.

It takes the best parts of SQL, Pandas and GraphQL, and combines them into a single Typescript-like language that compiles into SQL at run-time.

* Has the performance and abilities of SQL, and much more (* performance still needs work)
* Programmer friendly syntax and semantics, inspired by Python and Javascript
* Human-friendly errors, with gradual type-checking
* Interface through Python, HTTP or Repl

# Alternative to SQL

* Client-side first-order functions
* Better errors
* Saner syntax and type system

SQL() escape-hatch

# Alternative to GraphQL

Appropriate when using a single SQL database

# Alterative to an ORM


# Limitations

Limited by capabilities of database engine, performance & features

SQL() escape-hatch allows smart utilization of plugins


# Plans

* Mongo backend
* Support for multiple concurrent backends
* Clever query optimizations


-------------

Performance penalty is constant for each query

You always want your data to be as annotated as possible


Lists are tables:
 >> ["a", "b", "c"]{
        new_value: SQL(int, "upper($value) || $value")
    }
table list_string_proj120, count=3
new_value
-----------
Aa
Bb
Cc


# Notes

* Reqruires Python 3.8+

* Preql, PreQL, or PQL?