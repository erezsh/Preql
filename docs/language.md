# Language Reference

## Syntax

Inspired by Javascript

- Comments start with `//`
- Functions are defined with `func`

### Literals

Numbers are written as integers or floats.

```javascript
 >> type(10)
int
 >> type(3.14)
float
```

Operations between ints and floats result in a float:
```javascript
 >> type(10 + 3.14)
float
```

Division always returns a float. For "floordiv", use the `/~` operator:
```javascript
 >> 10 / 3
3.3333333333333335
 >> 10 /~ 3
3
```

Null values are specified with `null`. Null is only ever equal to itself:
```javascript
 >> null == null
true
```

Lists can be specified using the `[item1, item2, ...]` syntax. They are equivalent to a table with a single `item` column.

```javascript
 >> ["a", "b", "c"]
table  =3
┌───────┐
│ item  │
├───────┤
│ a     │
│ b     │
│ c     │
└───────┘
```

Ranges can be specified using the `[start..end]` syntax. They are equivalent to a list of numbers.

```javascript
 >> type([1..10])
list[int]
```

### Keywords

table
struct
func .. return
if .. else
while
import
throw
try .. catch
one
new

### Operators

| Category      | Operator      | Meaning  | Operates on |
| ------------- |:-------------:| -------- | ----- |
| Arithmetic    | | |
|               | + | Add      | Numbers, Strings, Lists, Tables
|               | - | Substract | Numbers
|               | * | Multiply | Numbers
|               | / | Divide | Numbers
|               | /~ | Divide into integer | Numbers
|               | % | Modulo | Numbers
|               | & | Element-wise 'and' | Lists, Tables
|               | | | Element-wise 'or' | Lists, Tables
|               | ^ | Element-wise 'xor' | Lists, Tables
| Comparison    | | |
|               | in | Find if value exists      | Primitive in Lists, Tables
|               | !in | Find if value does not exist | Primitive in Lists, Tables
|               | == | Equal | Any
|               | != | Not equal | Any
|               | <> | Not equal | Any
|               | < | Less than | Numbers
|               | <= | Less than or equal | Numbers
|               | > | Greater than | Numbers
|               | >= | Greater than or equal | Numbers
|               | and | Logical 'and' (lazy evaluation) | Any
|               | or | Logical 'or' (lazy evaluation) | Any
|               | not | Logical 'not' | Any
| Table Operations | | |
|                  | [] | Filter (where) | Lists, Tables
|                  | {} | Project (select) | Lists, Tables
|                  | [a..b] | Slice (limit, offset) | Lists, Tables
| Other | | |
|                  | () | Call Function | Functions
|                  | = | Assign value | Any
|                  | += | Add items | Collections
|                  | [..] | Slice | Strings, Collections
|                  | new | Create a new row in table | Tables
|                  | new[] | Create new rows in table | Tables



## Built-ins

### Types


| Category | Type | Supertype | Parameters | Meaning |
| - |  ---- | --------- | ---------- | ------- |
| Meta | | | |
| |  any  | | | Meta-type that is a superclass all other types |
| |  object  | | | Base object type |
| |  type  | | | The type of every type |
| Primitives | | | |
| | primitive | object | | Base primitive type |
| | bool  | primitive | |
| | number | primitive | |
| | int  | number | |
| | float | number | |
| | decimal | number | |
| | text  | primitive | |
| | string  | text | |
| | datetime  | primitive | |
| | t_id  | int | | id of table |
| | t_relation  | | int | refers to id of table |
| Containers | | | |
| |  container  | object | | Base container type |
| |  struct  | container | | |
| |  row  | struct | | Used for rows of tables |
| Collections | | | |
| | collection  | container | | Base collection type |
| | table  | collection | column types | |
| | list  | table | element type | |
| | set  | table | element type | |
| | vectorized  | collection | column types | Result of projection {} |
| | aggregate  | collection | column types | Result of aggregation {=>} |
| Misc | | | |
| | function  | object | |
| | module  | object | |
| | signal  | object | |
| | exception  | signal | |
| | nulltype  | | | type(null) |

See also:
* `type()`
* `isa()`
* `issubclass()`


### Functions

| Category | Signature | Description |
| - |  - | - |
| Interpreter |
| | `exit()` | Exit the current interpreter
| | `debug()` | Enter debug session (breakpoint)
| | `c()`, `continue()` | Resume from breakpoint (only available in debug session)
| | `continue()` | Resume from breakpoint (only available in debug session)
| | `dir()`, `names()` | Return a list of names in the current namespace.
| | `dir(obj)`, `names(obj)` | Return a list of names in the namespace of `obj`.
| | `help()` | Display help
| | `help(obj)` | Display help for `obj`
| Types |
| | `isa(inst, type)` | Test if `inst` is an instance of `type`, or one of its subtypes
| | `issubclass(type1, type2)` | Test if `type1` is a subclass of `type2`
| | `type(obj)` | Return the type of `obj`
| Escape Hatch |
| | `SQL(type, code)` | Executes `code` as SQL, and import the result as type `type`.
| | `PY(type, code)` | Executes `code` as Python, and import the result as type `type`.
| Database |
| | `connect(uri)` | Set up new database connection to URI (discards the existing session!)
| | `get_db_type()` | Returns a string for the database type (Currently: `"sqlite"` or `"postgres"`)
| Strings |
| | `upper(s)` | Uppercase
| | `lower(s)` | Lowercase
| | `length(s)` | String length
| Tables |
| | `count(col)` | Count the size of the collection (amount of rows / elements)
| | `count()` | Count how many aggregated rows there are
| | `count(agg_field)` | Count the size of the aggregated field vector (not including nulls)
| | `sum(agg_field)` | Sum the aggregated field (for vectors of `number`)
| | `mean(agg_field)` | Mean average of the aggregated field (for vectors of `number`)
| | `min(agg_field)` | Minimum of the aggregated field (for vectors of `number`)
| | `max(agg_field)` | Maximum of the aggregated field (for vectors of `number`)
| | `first(agg_field)` | First element of the aggregated field
| | `enum(table)` | Return a new table with an `index` column (starting from 0)
| | `concat(a, b)` | Conctenate the collections `a` and `b` (like `a+b`)
| | `intersect(a, b)` | Intersect the collections `a` and `b` (like `a&b`)
| | `union(a, b)` | Union the collections `a` and `b` (like `a|b`)
| | `substract(a, b)` | Substract the collections `a` and `b` (like except)
| | `temptable()`
| | `join(t1: a, t2: b)` | Join tables `a` and `b` into a new table of two structs: `t1` and `t2` (inner join)
| | `leftjoin(t1: a, t2: b)` | Join tables `a` and `b` into a new table of two structs: `t1` and `t2` (left join)
| | `joinall(t1: a, t2: b)` | Cartesian multiplication of tables `a` and `b` into a new table of two structs: `t1` and `t2`
| | `sample_ratio_fast(tbl, ratio)` | Return a random sample of rows from the table, at the approximate amount of (ratio*count(tbl))
| | `sample_fast(tbl, n, bias=0.05)` | Return a random sample of n rows from the table in one query (or at worst two queries)
| Misc |
| | `repr(obj)` | Returns a text representing the object
| | `cast(obj, type)` | Cast `obj` and return a new object of type `type`
| | `import_csv(table, filename)` | Improve rows from csv file `filename` into `table`
| | `random()` | Returns a random number between 0..1
| | `round(num)` | Returns a rounded float of `num`.
| | `pi()` | Returns PI
| | `now()` | Returns a datetime object for now

