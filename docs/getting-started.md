# Getting Started

## Install

Ensure you have, or install, Python 3.8 or up.

Install using pip:

```sh
    pip install -U prql
```

## Run interpreter

```sh
    preql
```

Preql will use Sqlite's memory database by default.

## Explore an existing database

```sh
    # Postgresql
    preql postgres://user:pass@host/dbname

    # MySQL
    preql mysql://user:pass@host/dbname

    # Sqlite
    preql sqlite://path/to/file'
```

When inside the Preql interactive prompt:

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