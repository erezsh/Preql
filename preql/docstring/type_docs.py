from preql.core.pql_types import T

DOCS = {
T.any: """A meta-type that can match any type.

	Examples:
		>> isa(my_obj, any)		// always returns true
		true
		>> isa(my_type, any)	// always returns true
		true
""",

# T.unknown: None,	# TODO

T.union: """A meta-type that means 'either one of the given types'

	Example:
		>> int <= union[int, string]
		true
		>> union[int, string] <= int
		false
		>> union[int, string] <= union[string, int]
		true

""",

T.type: """The type of types

	Examples:
		>> type(int) == type(string)
		true
		>> int <= type(int)			# int isn't a subtype of `type`
		false
		>> isa(int, type(int))		# int is an instance of `type`
		true
""",
T.object: """The base object type
""",

T.nulltype: """The type of the singleton `null`.

	Represents SQL `NULL`, but behaves like Python's `None`, 

	Examples:
		>> null == null
		true

		>> null + 1
		TypeError: Operator '+' not implemented for nulltype and int
""",

T.primitive: "The base type for all primitives",

T.text: "A text type (behaves the same as `string`)",
T.string: "A string type (behaves the same as `text`)",
T.number: "The base type for all numbers",
T.int: "An integer number",
T.float: "A floating-point number",
T.bool: "A boolean, which can be either `true` or `false`",
# T.decimal: "A decimal number",

T.datetime: "A datetime type (date+time combined)",
T.timestamp: "A timestamp type (unix epoch)",

T.container: """The base type of containers.

	A container holds other objects inside it. 
""",

T.struct: "A structure type",

T.row: "A row in a table. (essentially a named-tuple)",

# T.collection: """The base class of collections.

# 	A collection holds an array of other objects inside it.
# """,

T.table: """A table type.

	Tables support the following operations -
	- Projection (or: map), using the `{}` operator
	- Selection (or: filter), using the `[]` operator
	- Slice (or: indexing), using the `[..]` operator
	- Order (or: sorting), using the `order{}` operator
	- Update, using the `update{}` operator
	- Delete, using the `delete[]` operator
	- `+` for concat, `&` for intersect, `|` for union
""",

T.list: """A list type""",
T.set: """A set type, in which all elements are unique""",
T.projected: """A meta-type to signify projected operations, i.e. operations inside a projection.

	Example:
		>> x = [1]
		>> one one x{ repr(type(item)) }
		"projected[item: int]"
""",
T.aggregated: """A meta-type to signify aggregated operations, i.e. operations inside a grouping

	Example:
		>> x = [1]
		>> one one x{ => repr(type(item))}
		"aggregated[item: int]"
""",
T.t_id: "The type of a table id",
T.t_relation: "The type of a table relation",


T.json: "A json type",
T.json_array: "A json array type. Created by aggregation.",

T.function: "A meta-type for all functions",
T.module: "A meta-type for all modules",
T.signal: "A meta-type for all signals (i.e. exceptions)",

}
