# Features

Preql is a programming language, a library, an interactive shell, and a set of tools.

## Programming Language

- **Modern syntax and semantics**
    - Interpreted, everything is an object
    - Strong type system with gradual type validation and duck-typing
    - Modules, functions, exceptions, tables, structs
- **SQL integration**
    - Compiles to SQL whenever possible (guaranteed for all table operations)
    - Escape hatch to SQL (write raw SQL expression within Preql code)
    - Support for multiple SQL targets
        - **Sqlite**
        - **Postgres**
        - **MySQL**
        - Askgit :)
        - BigQuery (soon)
        - More to come!

- **Python integration**
    - Use from Python as a library
    - Call Python from Preql
    - Pandas integration

- **Interactive Environment**
    - Shell (REPL), with auto-completion
    - Runs on Jupyter Notebook, with auto-completion

- REST+JSON server, automatically generated


## Planned features

- See the [roadmap](roadmap.md)