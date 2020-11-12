# Getting Started

## Install

Ensure you have, or install, Python 3.8 or up.

Install using pip:

```sh
    pip install -U prql
```

## Run interpreter in the console

```sh
    preql
```

Preql will use Sqlite's memory database by default.

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


## Explore an existing database

When you start the interpreter, you can specify which database to connect to.

```sh
    # Postgresql
    preql postgres://user:pass@host/dbname

    # MySQL
    preql mysql://user:pass@host/dbname

    # Sqlite
    preql sqlite:///path/to/file'
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

## Further reading

- [Learn the language](language.md)
- [Read the tutorials](tutorial.md)