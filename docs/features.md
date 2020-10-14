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
        - More to come!

- **Python integration**
    - Use from Python as a library
    - Call Python from Preql
    - Pandas integration

- **Interactive shell** (REPL) with autocompletion
- REST+JSON server, automatically generated


## Planned features (for near future)
- Full IDE support via language server (WIP, coming soon)
- Support for more databases
- Automatic GraphQL integration
- Multiple Dispatch (multimethods)
- Cached JIT compilation (PoC working!)
- JSON operations in SQL
- Compile control flow to SQL
- API for Graph computation over SQL
- Migrations
- Automatic joins via attribute access