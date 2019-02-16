# First things first

- Do autojoins

- Use method syntax?
    Table order(name) :limit(3) 
    Table groupby(name, date) order(-date)
    Table :offset(2)
    Table :range(1..2)
    Table :page(1, 40)

- Possible groupby syntax:
    Table {name, date ==> count(id), some_aray}
    Table {count(id), some_array foreach name, date}

# More stuff

- Many-to-many

- Auto-Join via connecting table? (i.e. graph traverse). Or just for m2m

- GraphQL like syntax into JSON

- Private/public tables/members/functions

- Validation

- ORM for results?

- Auto-limit?

- Support index, check, default, unique

- Store schema in meta table

- Lift query selections / projections when possible (for example through a join)

- Import

- Better validation and error messages

- Offset without limit

- Fix bug1.pql

- Automatically deduce tables? Maybe use SQLAlchemy for it?

# Done Already
- Named joins to support recursion and other stuff
- Two-step: Bind + type inference -> generate sql
- Explicit joins
- Order by !!
- Offset + limit
