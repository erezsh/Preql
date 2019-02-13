# Preql
Pretty Query Language

(This is a placeholder repo. I will upload a project once I'm settled on a design)

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
