"""This module provides functions for importing the return values from SQL queries.

The value are already "native python", but not conforming to the expected type.

We perform two operations:
- Place the items into the expected structure (instead of just a flat list)
- Convert primitives to the expected type

"""

from datetime import datetime
import decimal
import json

import arrow

from preql.utils import listgen, safezip
from .pql_types import T, dp_type, dp_inst
from .exceptions import Signal
from .types_impl import flatten_type
from preql.context import context
from .sql import _ARRAY_SEP, sqlite, mysql


def _from_datetime(s):
    if s is None:
        return None

    # Postgres
    if isinstance(s, datetime):
        return s

    # Sqlite
    if isinstance(s, str):
        try:
            # return datetime.fromisoformat(s)
            return arrow.get(s)
        except ValueError as e:
            raise Signal.make(T.ValueError, None, str(e))

    raise Signal.make(T.TypeError, None, f"Unexpected type for datetime: {type(s)}")


@dp_type
def _restructure_result(t, i):
    raise Signal.make(T.TypeError, None, f"Unexpected type used: {t}")

@dp_type
def _restructure_result(t: T.table, i):
    # return ({name: _restructure_result(state, col, i) for name, col in t.elem_dict.items()})
    return next(i)

@dp_type
def _restructure_result(t: T.struct, i):
    return ({name: _restructure_result(col, i) for name, col in t.elems.items()})

@dp_type
def _restructure_result(_t: T.union[T.primitive, T.nulltype], i):
    return _from_sql_primitive(next(i))


@dp_type
def _restructure_result(t: T.json_array[T.union[T.primitive, T.nulltype]], i):
    res = next(i)
    if not res:
        return res

    target = context.state.db.target
    if target == mysql:
        res = json.loads(res)
    elif target == sqlite:
        if not isinstance(res, str):
            raise Signal.make(T.TypeError, None, f"json_array type expected a string separated by {_ARRAY_SEP}. Got: '{res}'")
        res = res.split(_ARRAY_SEP)

    # XXX hack! TODO Use a generic form to cast types
    try:
        if t.elem <= T.int:
            res = [int(x) for x in res]
        elif t.elem <= T.float:
            res = [float(x) for x in res]
    except ValueError:
        raise Signal.make(T.TypeError, None, f"Error trying to convert values to type {t.elem}")

    return res

@dp_type
def _restructure_result(_t: T.datetime, i):
    s = next(i)
    return _from_datetime(s)

@dp_type
def _restructure_result(_t: T.timestamp, i):
    s = next(i)
    return _from_datetime(s)



def _extract_primitive(res, expected):
    try:
        row ,= res.value
        item ,= row
    except ValueError:
        raise Signal.make(T.TypeError, None, f"Expected a single {expected}. Got: '{res.value}'")

    return _from_sql_primitive(item)

@dp_inst
def sql_result_to_python(res: T.bool):
    item = _extract_primitive(res, 'bool')
    if item not in (0, 1):
        raise Signal.make(T.ValueError, None, f"Expected SQL to return a bool. Instead got '{item}'")
    return bool(item)

@dp_inst
def sql_result_to_python(res: T.int):
    item = _extract_primitive(res, 'int')
    if not isinstance(item, int):
        raise Signal.make(T.ValueError, None, f"Expected SQL to return an int. Instead got '{item}'")
    return item

@dp_inst
def sql_result_to_python(res: T.primitive):
    return _extract_primitive(res, res)

@dp_inst
def sql_result_to_python(res):
    return res.value

@dp_inst
def sql_result_to_python(res: T.datetime):
    # XXX doesn't belong here?
    item = _extract_primitive(res, 'datetime')
    return _from_datetime(item)

@dp_inst
def sql_result_to_python(res: T.timestamp):
    # XXX doesn't belong here?
    item = _extract_primitive(res, 'datetime')
    return _from_datetime(item)

def _from_sql_primitive(p):
    if isinstance(p, decimal.Decimal):
        # TODO Needs different handling when we expect a decimal
        return float(p)
    elif isinstance(p, bytearray):
        return p.decode()
    return p

@dp_inst
def sql_result_to_python(arr: T.list):
    fields = flatten_type(arr.type)
    if not all(len(e)==len(fields) for e in arr.value):
        raise Signal.make(T.TypeError, None, f"Expected 1 column. Got {len(arr.value[0])}")

    if arr.type.elem <= T.struct:
        return [{n: _from_sql_primitive(e) for (n, _t), e in safezip(fields, tpl)} for tpl in arr.value]
    else:
        return [_from_sql_primitive(e[0]) for e in arr.value]

@dp_inst
@listgen
def sql_result_to_python(arr: T.table):
    expected_length = len(flatten_type(arr.type))   # TODO optimize?
    for row in arr.value:
        if len(row) != expected_length:
            raise Signal.make(T.TypeError, None, f"Expected {expected_length} columns, but got {len(row)}")
        i = iter(row)
        yield {name: _restructure_result(col, i) for name, col in arr.type.elems.items()}





def _bool_from_sql(n):
    if n == 'NO':
        n = False
    elif n == 'YES':
        n = True
    assert isinstance(n, bool), n
    return n

def type_from_sql(type, nullable):
    type = type.lower()
    d = {
        'integer': T.int,
        'int': T.int,           # mysql
        'tinyint(1)': T.bool,   # mysql
        'serial': T.t_id,
        'bigserial': T.t_id,
        'smallint': T.int,  # TODO smallint / bigint?
        'bigint': T.int,
        'character varying': T.string,
        'character': T.string,  # TODO char?
        'real': T.float,
        'float': T.float,
        'double precision': T.float,    # double on 32-bit?
        'boolean': T.bool,
        'timestamp': T.timestamp,
        'timestamp without time zone': T.timestamp,
        'timestamp with time zone': T.datetime,
        'datetime': T.datetime,
        'date': T.date,
        'time': T.time,
        'text': T.text,
    }
    try:
        v = d[type]
    except KeyError:
        if type.startswith('int('): # TODO actually parse it
            return T.int
        elif type.startswith('tinyint('): # TODO actually parse it
            return T.int
        elif type.startswith('varchar('): # TODO actually parse it
            return T.string

        return T.string.as_nullable()

    nullable = _bool_from_sql(nullable)

    return v.replace(_nullable=nullable)
