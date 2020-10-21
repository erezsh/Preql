API Reference
=============

Preql
-----

.. autoclass:: preql.Preql
    :members: __init__, load, import_pandas, load_all_tables, start_repl

.. autoclass:: preql.api.TablePromise
    :members: to_json, to_pandas, __len__, __eq__, __getitem__, __iter__

.. autoexception:: preql.exceptions.Signal
    :members: get_rich_lines, __str__, repr