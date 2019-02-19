# Preql
Pretty Query Language

(This project is still under works. Stay tuned as I approach a working alpha)

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