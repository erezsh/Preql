# Preql

**Preql** (stands for: Pretty Query Language) is an interpreted, typed, functional query language that compiles to SQL.

(Preql is currently at the pre-alpha stage, and isn't ready for real-world use yet)

Preql aims to provide the most common and necessary features for working with real-world relational data, with clean syntax and code that's easy to work with.

## But why?

Modern SQL databases offer an amazing set of features for working with complex data. They're fast, reliable, and support a wide range of useful operations and features.

But at the same time, the interface to these features, which is the SQL language, is stuck in the 80's, somewhere along the developmental phase of QBasic.

Even "new" SQL still has an incredibly clunky syntax, a confusing standard library, and no support for real abstractions, such as objects or first-class functions. It also looks very different for every database, making portability nearly impossible.

Preql attempts to fix this, with the following features:
- Modern syntax and semantics, inspired by Python and other contemporary languages
- First-class functions
- Session variables
- An object model, with automatic table joins based on definition (optional)
- Portable - can compile to different dialects, based on the target database (future feature)
- Simple interface for use with REST or other programming languages

Note: Preql doesn't intend to support every SQL feature out there. It aims to provide the popular and most-used features, and allow embedded SQL for the edge-cases.

## Example

The following code examples are all implemented and working already.

```ruby
        #
        # Table definitions
        #
        table Country:
            name: string
            language: string

        table Person:
            name: string
            age: integer?
            country: Country -> citizens         # Define a foreign-key with a backref

        #
        # Inserts
        #
        add Country("United States", "en") as us
        add Country("England", "en") as england         # Save as variable
        add Country("France", "fr") as france
        add Person("George Orwell", country=england)    # Use the variable whenever you like
        ...

        #
        # Query definitions (i.e defines local functions)
        #
        adults = Person [age >= 18]         # Square-brackets create a filter
        count_adults = adults.count()     # Construct queries using other queries
        others_from_my_country = Person [country=me.country, id!=me.id]     # More variable use

        # Auto-join examples
        adults_with_country = adults {name, country.name}   # Curly-braces choose attributes
        english_speakers = Person [country.language = "en"] {id, name, country.name}

        # Group-by examples
        population_count = Country {name => count(citizens)}
        citizens_list = Country {name => citizens.name}     # Creates an array of Person names

        # Order-by & limit examples
        youngest_first = Person order(age)
        oldest_person = Person order(-age).limit(1)

        # Example of an explicit automatic join. Equivalent to: english_speakers
        english_speakers__explicit_join = (
            join(c = Country[language="en"], p = Person)
            { p.id, p.name, c.name }
        )

        # Example of a free join (like in SQL). Equivalent to: english_speakers
        english_speakers___free_join = (
            freejoin(c = Country, p = Person)
            [c.language="en", p.country = c]
            { p.id, p.name, c.name }
        )



```


## Who is this for?

### Web developers

### Data Scientists

### Data Engineers