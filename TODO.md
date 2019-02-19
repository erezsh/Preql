# First things first

- Allow updating of rows

- Use method syntax?
    Table order(name) :limit(3) 
    Table groupby(name, date) order(-date)
    Table :offset(2)
    Table :range(1..2)
    Table :page(1, 40)

- Possible groupby syntax:
    Table {name, date ==> count(id), some_aray}
    Table {count(id), some_array foreach name, date}

- Many-to-many

# More stuff

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

- Add support for procedural constructs (for, if, etc.)

- var name = query  creates a temporary table, for efficiency


# Done Already
- Named joins to support recursion and other stuff
- Two-step: Bind + type inference -> generate sql
- Explicit joins
- Order by !!
- Offset + limit
- Do autojoins
