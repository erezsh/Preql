# Getting Started

## Install

Ensure you have, or install, Python 3.8 or up.

Install using pip:

```sh
    pip install -U prql
```

## Run the interpreter in the console (REPL)

After installation, just enter:

```sh
    preql
```

Preql will use Sqlite's memory database by default.

### Explore an existing database

When you start the interpreter, you can specify which database to connect to.

```sh
    # Postgresql
    preql postgres://user:pass@host/dbname

    # MySQL
    preql mysql://user:pass@host/dbname

    # Sqlite (use existing or create new)
    preql sqlite:///path/to/file
```

When already inside the Preql interactive prompt, or in a Jupyter Notebook, use the `connect()` method.:

```go
    connect("sqlite:///path/to/file")
```

Use introspective methods to see a list of the tables, and of the available functions:

```go
    // Get a list of all tables in database
    >> tables()

    // Get help regarding how to use Preql
    >> help()

    // For example:
    >> help(connect)

    func connect(uri: string) = ...

        Connect to a new database, specified by the uri
```

### See the SQL

If you want to see the SQL code that is being executed, you can run Preql with `preql --print-sql`.

It can be useful for debugging, but isn't recommended for regular workflow.

## Run in Jupyter Notebook

First you need to install the plugin, using the following command:

```sh
    preql --install-jupyter
```

Then just run Jupyter Notebook as usual:
```sh
    jupyter notebook
```

And create a new notebook with the `Preql` kernel.

Use the `connect()` function to connect to a database.

## Use as a Python library

```python
from preql import Preql
p1 = Preql()                             # Use memory database
p2 = Preql("sqlite:///path/to/file")     # Use existing or new file

assert p1('sum([1..10])') == 45
```

## Run as a REST / GraphQL server

Coming soon!

## Further reading

- [Learn the language](language.md)
- [Read the tutorial](tutorial.md)