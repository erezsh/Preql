# Preql
Pretty Query Language

(This is a placeholder repo. I will upload a project once I'm settled on a design)

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

        adults = Person(age >= 18)
        adults_with_country = adults {id, country.name}
        some_query = Person(age < 50, country.name = "United States") {id,name}
```
