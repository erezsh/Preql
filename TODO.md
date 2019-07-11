# First things first

- Explicit joins
- Named joins to support recursion and other stuff
- Order by + asc/desc
- Offset + limit
- Do autojoins

- Allow updating of rows

- Use method syntax
    Table .offset(2)
    Table .range(1..2)
    Table .page(1, 40)

- Many-to-many

- date fields

- Always include id by default in projection?
    - Pros: Makes autojoin & ORM usually work
    - Cons: May be confusing

- Throw error on duplicate projection names

# More stuff

- Differenciate between join and left-join according to ! or ?

- smart not-in ("in" is a simple join, but "not in" requires a separate query)

- auto rename sql-keywords (to allow column names like "group")

- Table.count() = count(self)

- Enums

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
