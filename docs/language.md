# Language Reference

(This document is incomplete, and needs more work)

## Syntax

Inspired by Javascript

- Comments start with `//`

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

### Functions

- Functions are defined with `func`, like in Go

```javascript
func add1(x) = x + 1        // short-hand syntax

func abs(x) {
    if (x < 0) {
        return -x
    }
    return x
}
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

