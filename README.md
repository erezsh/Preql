# Preql

**Preql** (stands for: Pretty Query Language) is an interpreted, typed, functional query language that compiles to SQL.

(Preql is currently at the pre-alpha stage, and isn't ready for real-world use yet)

## Why?

Modern SQL databases offer an amazing set of features for working with complex data. They're fast, reliable, and support a wide range of useful operations and features.

But at the same time, the interface to these features, which is the SQL language, is stuck in the 80's, somewhere along the developmental phase of QBasic.

Even "new" SQL still has an incredibly clunky syntax, a confusing standard library, and no support for real abstractions, such as objects or first-class functions. It also looks very different for every database, making portability nearly impossible.

Preql attempts to fix this. Here's some of the features that it offers:
- Modern syntax and semantics, inspired by Python and others
- First-class functions
- Automatic table joins based on table definitions
- Portable - can compile to different dialects, based on the target database (future feature)

Note: Preql doesn't intend to support every SQL feature out there. It aims to provide the popular and most-used features, and allow embedded SQL for the edge-cases.

## Example

```ruby
        # Table definitions
        table Country:
            name: string
            language: string

        table Person:
            name: string
            age: integer?
            country: Country -> citizens         # Define a foreign-key with backref

        # Inserts
        add Country("England", "en") as england
        add Country("United States", "en") as us
        add Country("France", "fr") as france
        add Person("George Orwell", country=england)
        ...

        # Query definitions
        adults = Person [age >= 18]
        adults_with_country = adults {name, country.name}
        english_speakers = Person [country.language = "en"]
        population_count = Country {name, count(citizens)}
```



##

Preql has a different emphasis than SQL. That comes into play in a few ways:

Preql has features that SQL doesn't, but also it isn't trying to provide a feature-complete alternative to SQL. The goal is to provide the most common and necessary features for working with real-world relational data, with clean syntax and code that's easy to work with.

For example, while Preql supports nested queries, its syntax isn't optimized for that. Instead, the recommended style is to create named queries for each part, and then to combine them incrementally to create the full query.


## Who is this for?

### Web developers

### Data Scientists

### Data Engineers