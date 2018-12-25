# Preql
Pretty Query Language

(This project is still under works. Stay tuned as I approach a working alpha)

## Example

```ruby
        table Country:
            name: str?

        table Person:
            name: str
            age: int?
            country -> Country

        add Country(name="England") as england
        add Country(name="United States")
        add Person(name="Orwell", country=england)

        adults = Person[age >= 18]
        adults_with_country = adults [id, country.name]
        some_query = Person [age < 50, country.name = "United States"] {id,name}
```



##

Preql has a different emphasis than SQL. That comes into play in a few ways:

Preql has features that SQL doesn't, but also it isn't trying to provide a feature-complete alternative to SQL. The goal is to provide the most common and necessary features for working with real-world relational data, with clean syntax and code that's easy to work with.

For example, while Preql supports nested queries, its syntax isn't optimized for that. Instead, the recommended style is to create named queries for each part, and then to combine them incrementally to create the full query.