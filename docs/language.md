# Language Reference

(This document is incomplete, and needs more work)

Preql's syntax is a mix between Go and Javascript.

- Comments start with `//`

## Literals

### Null

Null values are specified with `null`. Null is only ever equal to itself:
```javascript
 >> null == null
true
```

### Booleans

The two boolean values are `true` and `false`, and both are of type `bool`.

```javascript
 >> type(true)
bool
 >> type(false)
bool
````

### Numbers

Numbers are written as integers or floats.

Standard operators apply: `+`, `-`, `*`, `/`, `%`, `**`, `==`, `!=`, `<`, `>`, `<=`, `>=`

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


### Strings

Standard operators apply: `+`, `*`, `==`, `!=`

Strings, like in Python, take one of the following forms:

- `'a'`
- `"a"`
- `'''a'''`
- `"""a"""`

```javascript
 >> type("a")
string
```

Triple-quoted strings capture newlines, while single-quoted strings do not.

Strings support the `like` operator, or `~`, similar to SQL's like operator:

```javascript
 >> "hello" ~ "h%"
true
```

Strings support the slicing operators `[]`:

```javascript
 >> "preql"[3..]
"ql"
```

### Structs

Structs can be created on the fly, using the `{}` syntax:

Structs are essentially dictionaries (or maps), in which the keys are always of the type string.

```javascript
 >> x = {a:1, b:2}
{a: 1, b: 2}
 >> x.a
1
 >> type(x)
struct[a: int, b: int]
```

### Lists

Lists can be specified using the `[item1, item2, ...]` syntax. They are equivalent to a table with a single `item` column.

Lists support all tables operations.

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

### Ranges

Ranges can be specified using the `[start..end]` syntax. They are equivalent to a list of numbers.

```javascript
 >> type([1..10])
list[int]
```

### Functions

- Functions are defined with `func`, like in Go

```javascript
// normal syntax
func abs(x) {
    "docstring"
    if (x < 0) {
        return -x
    }
    return x
}

// short-hand syntax
func add1(x) = x + 1
    "docstring"
```

## Keywords

| Category     | Keyword   | Meaning  |
| ------------ |:---------:| -------- |
| Definitions  | | |
|              | table | Define a new table |
|              | struct | Define a new struct |
|              | func | Define a new function |
|              | const | Modifier for definition, signifying a constant (immutable) |
| Control Flow | | |
|              | if, else | Conditional |
|              | while, for, in | Loop |
|              | try, catch, throw | Exception handling |
| Operators        | | |
|              | new, one | See below |
|              | and, or, not | See below |
| Other        | | |
|              | import | Imports a module |
|              | assert | Assert a condition or throw an exception |
|              | print | Prints to stdout |
|              | null, false, true | Constant literals |


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
|               | \| | Element-wise 'or' | Lists, Tables
|               | ^ | Element-wise 'xor' | Lists, Tables
| Comparison    | | |
|               | in | Find if value exists      | Primitive in Lists, Tables
|               | !in | Find if value does not exist | Primitive in Lists, Tables
|               | ~, like | 'Like' pattern matching | Strings
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
|                  | order{} | Order | Lists, Tables
|                  | update{} | Update | Tables
|                  | delete{} | Delete | Tables
| Other | | |
|                  | () | Call Function | Functions
|                  | = | Assign value | Any
|                  | += | Add items | Collections
|                  | [..] | Slice | Strings, Collections
|                  | new | Create a new row in table | Tables
|                  | new[] | Create new rows in table | Tables
|                  | one | Returns one item, or throws an exception  | Lists, Tables
