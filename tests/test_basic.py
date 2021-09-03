from multiprocessing.pool import ThreadPool

from preql.core.sql import mysql, bigquery, sqlite
from unittest import skip, SkipTest

from parameterized import parameterized_class

from preql.core.pql_objects import UserFunction
from preql.core.exceptions import Signal
from preql.core.pql_types import T, Id
from preql.sql_interface import _drop_tables

from .common import PreqlTests, SQLITE_URI, POSTGRES_URI, MYSQL_URI, DUCK_URI, BIGQUERY_URI

def uses_tables(*names):
    names = [n for n in names]
    def decorator(decorated):
        def wrapper(self):
            if self.uri not in (MYSQL_URI, BIGQUERY_URI):
                return decorated(self)

            p = self.Preql()
            def _key(t):
                try:
                    return names.index(t.lower())
                except ValueError:
                    return -1
            tables = sorted(p._interp.list_tables(), key=_key)
            # if tables:
            #     print("@@ Deleting", tables, decorated)
            _drop_tables(p._interp.state, *tables)
            # assert not tables
            try:
                return decorated(self)
            finally:
                p = self.preql
                # Table contents weren't dropped, due to autocommit
                if p._interp.state.db.target in (mysql, bigquery):
                    _drop_tables(p._interp.state, *map(Id,names))
                tables = p._interp.list_tables()
                assert not tables, tables

        return wrapper
    return decorator

def is_eq(a, b):
    a = [tuple(row.values()) for row in a]
    return a == b

NORMAL_TARGETS = [
    ("Normal_Lt", SQLITE_URI, True),
    ("Normal_Pg", POSTGRES_URI, True),
    ("Normal_My", MYSQL_URI, True),
    ("Normal_Bq", BIGQUERY_URI, True),
    # ("Normal_Dk", DUCK_URI, True),
]
UNOPTIMIZED_TARGETS = [
    ("Unoptimized_Lt", SQLITE_URI, False),
    ("Unoptimized_Pg", POSTGRES_URI, False),
]

@parameterized_class(("name", "uri", "optimized"), NORMAL_TARGETS + UNOPTIMIZED_TARGETS)
class BasicTests(PreqlTests):
    @uses_tables('Person', 'Country')
    def test_basic1(self):
        preql = self.Preql()
        preql.load('country_person.pql', rel_to=__file__)

        self._test_basic(preql)
        self._test_ellipsis(preql)
        self._test_ellipsis_exclude(preql)
        self._test_user_functions(preql)
        self._test_joins(preql)
        self._test_cache(preql)
        self._test_groupby(preql)
        self._test_temptable(preql)

    def _test_basic(self, preql):
        self.assertEqual(float(preql("3.14")), 3.14)    # cast to float, because postgres may return Decimal

        assert preql("1") == 1
        assert preql("-(1)") == -1
        assert preql("1 / 2") == 0.5
        assert preql("10 /~ 3") == 3
        assert preql("type(10/3) == float")

        # GroupBy will use the old value if TableTypes aren't versioned
        self.assertEqual( preql("[1,2,3]{v: item/~2 => sum(item)}").to_json(), [{'v':0, 'sum': 1}, {'v':1, 'sum':5}])
        self.assertEqual( preql("[1,2,3]{item: item/~2 => sum(item)}").to_json(), [{'item':0, 'sum': 1}, {'item':1, 'sum':5}])


        preql("""func query1() = list(Country[language=="en"]{name})""")

        assert set(preql.query1()) == {"England", "United States"}, preql.query1()

        res = preql("list(Person[country==isr]{name})")
        self.assertEqual(set(res), {"Erez Shinan", "Ephraim Kishon"})

        res = preql("list(Person[id!=me]{name})")
        self.assertEqual(set(res), {"Ephraim Kishon", "Eric Blaire", "H.G. Wells", "John Steinbeck"})

    def _test_cache(self, preql):
        # Ensure that names affect type
        self.assertEqual( list(preql('Person {name2: name}')[0].keys()), ['name2'])
        self.assertEqual( list(preql('Person {name}')[0].keys()) , ['name'])
        self.assertEqual( list(preql('Person {name2: name}')[0].keys()), ['name2'])

    def _test_ellipsis(self, preql):

        assert preql('Person {name, ...}[name=="Erez Shinan"]{name}') == [{'name': 'Erez Shinan'}]

        self.assertEqual( list(preql('Person {name, ...}')[0].keys()) , ['name', 'id', 'country'])
        assert list(preql('Person {country, ...}')[0].keys()) == ['country', 'id', 'name']
        self.assertEqual( list(preql('Person {..., id}')[0].keys()) , ['name', 'country', 'id'] )
        assert list(preql('Person {country, ..., id}')[0].keys()) == ['country', 'name', 'id']

        self.assertEqual( list(preql('Person {name2: name, ...}')[0].keys()), ['name2', 'id', 'country'])
        assert list(preql('Person {name2: name, ..., name3: name}')[0].keys()) == ['name2', 'id', 'country', 'name3']
        assert list(preql('Person {name: name, ...}')[0].keys()) == ['name', 'id', 'country']

        self.assertEqual( list(preql('Person {name2: name+"!", ...}')[0].keys()), ['name2', 'id', 'name', 'country'])
        self.assertEqual( list(preql('Person {name2: name+"!", ..., name3: name+"!"}')[0].keys()), ['name2', 'id', 'name', 'country', 'name3'])
        self.assertEqual( list(preql('Person {name2: name+"!", ..., name3: name}')[0].keys()), ['name2', 'id', 'country', 'name3'])

        self._assertSignal(T.SyntaxError, preql, 'Person {x: ...}')
        self._assertSignal(T.SyntaxError, preql, 'Person {...+"a", 2}')

    def _test_ellipsis_exclude(self, preql):
        self.assertEqual( preql('Person {name, ... !id !country}[name=="Erez Shinan"]'), [{'name': 'Erez Shinan'}] )

        assert list(preql('Person {name, ... !id !country}')[0].keys()) == ['name']
        assert list(preql('Person {country, ... !name}')[0].keys()) == ['country', 'id']
        assert list(preql('Person {... !name, id}')[0].keys()) == ['country', 'id']
        assert list(preql('Person {country, ... !name, id}')[0].keys()) == ['country', 'id']


        self._assertSignal(T.NameError, preql, '[3]{... !hello}')
        self._assertSignal(T.TypeError, preql, '[3]{... !item}')

        # TODO exception when name doesn't exist

    def test_empty_count(self):
        # TODO with nulls

        preql = self.Preql()
        assert preql("one one [1,2,3] { => count()} ") == 3
        assert preql(" [1,2,3] { item /~ 2 => count()} {count} ") == [{'count': 1}, {'count': 2}]

    def test_assert(self):
        preql = self.Preql()
        self._assertSignal(T.AssertError, preql, 'assert 0')

    def test_arith(self):
        preql = self.Preql()

        assert preql("1 + 2 / 4") == 1.5
        assert preql("1 + 2 /~ 4 + 1") == 2
        assert preql('"a" + "b"') == "ab"
        assert preql('"a" * 3') == "aaa"
        assert preql('"ab" * 3') == "ababab"
        self.assertEqual( preql('"a" + "b"*2 + "c"'), 'abbc' )
        assert preql('"a" ~ "a%"')
        assert preql('"abc" ~ "a%"')
        assert preql('"abc" ~ "a%c"')
        assert not preql('"ab" ~ "a%c"')

        assert preql("1 - 2**3") == -7
        assert preql("10**-2") == 0.01

        # res = preql("list([0, 10, 1.2]{item ** 2})")
        res = preql("list([0.0, 10.0, 1.2]{item ** 2})")
        assert res == [0.0, 100.0, 1.44], res


        self._assertSignal(T.TypeError, preql, '"a" + 3')
        self._assertSignal(T.TypeError, preql, '"a" ~ 3')
        self._assertSignal(T.TypeError, preql, '"a" - "b"')
        self._assertSignal(T.TypeError, preql, '"a" % "b"')
        self._assertSignal(T.TypeError, preql, '3 ~ 3')

    def test_table_arith(self):
        preql = self.Preql()

        assert preql("[1] + [2]") == [1, 2]
        assert preql("[1] + []") == [1]
        assert preql("[] + [1]") == [1]
        assert preql("[]") == []

    def test_foreign_key(self):
        p = self.Preql()

        p(r'''
            table Point {
                x: int
                y: int
            }

            table HRuler {
                x_axis: Point.x
            }

            new Point(1, 1)
            new Point(3, 3)
            new Point(3, 4)
        ''')

        p('join(h: HRuler[x_axis==3], p: Point) {p.y}') == [3, 4]
        p('join(h: HRuler[x_axis==4], p: Point) {p.y}') == []

    def test_logical(self):
        # TODO null values
        preql = self.Preql()
        assert preql('1==0 or isa(1, int)')
        assert not preql('1==0 or isa(1, float)')
        res = preql('[0,1,2,3][item < 2]{r: item or 0}')
        assert res == [{'r': 0}, {'r': 1}], res
        res = preql('[0,1,2,3][item < 2]{r: item or 10}')
        assert res == [{'r': 10}, {'r': 1}], res

        res = preql('[0,1,2,3]{r: item > 1 and item < 3}[r]')
        assert res == [{'r': 1}], res
        res = preql('[0,1,2,3]{r: item < 3, item}[not r]')
        assert res == [{'r': 0, 'item': 3}], res

        self.assertRaises(Signal, preql, '"hello" or 1') # XXX TypeError

        assert preql('"hello" or "a"') == "hello"
        assert preql('"hello" and "a"') == "a"
        assert preql('"hello" and ""') == ""
        assert preql('"" and "hello"') == ""
        assert preql('"" or "hello"') == "hello"
        assert preql('"" or "hello"') == "hello"
        assert preql('"bla" and "hello"') == "hello"

        assert preql('1 or 2 or 3') == 1
        assert preql('1 and 2 and 3') == 3
        assert preql('1 and 2 or 3') == 2
        assert preql('1 or 2 and 3') == 1

        self.assertEqual( preql('[1] or [2]'), [1])
        self.assertEqual( preql('[1] and [2]'), [2])

    def test_basic2(self):
        # More basic tests
        preql = self.Preql()

        z = preql('''
            x = [1..4]
            x[item == max(x)]
        ''')
        assert z == [3]

        try:
            assert preql('["a".."c"]') == ['a', 'b', 'c']
        except Signal:
            # TypeError for now
            pass

        self.assertRaises(Signal, preql, '[min..]')    # XXX TypeError

        assert preql('"hello"[1]') == 'e'
        res = preql('list(["hello"]{item[1..2]})')
        assert res == ["e"], res

        res = preql('list(["hello"]{item[1]})')
        assert res == ["e"], res

        self.assertRaises(Signal, preql, 'return 1')


    def test_from_python(self):
        p = self.Preql()

        p('func f(x) = count(x)')
        assert p.f([1,2,3]) == 3

        # TODO
        # assert p.f([(1,2), (2,3), (3,4)]) == 3
        assert p.count([1,2,3]) == 3
        assert p.enum([1]) == [{'index': 0, 'item': 1}]

        assert p.SQL(int, "SELECT 2") == 2
        assert p.SQL(p.int, "SELECT 2") == 2
        assert p.SQL(p.list[p.int], "SELECT 1 UNION ALL SELECT 2") == [1,2]
        assert p.SQL(p('list[int]'), "SELECT 1 UNION ALL SELECT 2") == [1,2]


    def test_vectorized_logic2(self):
        preql = self.Preql()
        res = preql('list(["a", "b"]{length(item)>1 or true})')
        assert res == [1, 1]

        assert preql(' ["hello"]{item[..1]} ') == [{'_': 'h'}]

        self.assertRaises(Signal, preql, '["hello"]{item or 1}')    # XXX TypeError

        res = preql('["hello"]{item or "a"}')
        assert res == [{'_': "hello"}], res
        res = preql('["hello"]{item and "a"}')
        assert res == [{'_': "a"}], res
        res = preql('["hello"]{item and ""}')
        assert res == [{'_': ""}], res
        res = preql('[""]{item or "a"}')
        assert res == [{'_': "a"}], res
        res = preql('[""]{item and "a"}')
        assert res == [{'_': ""}], res
        res = preql('[""]{item or ""}')
        assert res == [{'_': ""}], res
        res = preql('[""]{item and ""}')
        assert res == [{'_': ""}], res

        res = preql('["hello"]{item or "a" and "b"}')
        assert res == [{'_': "hello"}], res
        res = preql('["hello"]{item and "a" or "b"}')
        assert res == [{'_': "a"}], res

        res = preql('list(["a", "b"]{length(item)>1 or true})')
        assert res == [1, 1]



    @uses_tables('backup', 'Point')
    def test_update_basic(self):
        preql = self.Preql()
        preql("""
        table Point {x: int, y: int}

        new Point(1,3)
        new Point(2,7)
        new Point(3,1)
        new Point(4,2)

        const table backup = Point

        func p2() = Point[x==3] update{y: y + 13}
        func p() = p2() {...!id}
        """)
        assert preql.p() == [{'x': 3, 'y': 14}]
        assert preql.p() == [{'x': 3, 'y': 27}]
        assert preql('backup[x==3]{y}') == [{'y': 1}]
        res = preql('backup[x==3] update {y: x+y}')
        assert res == [{'id': res[0]['id'], 'x': 3, 'y': 4}], res
        assert preql('backup[x==3]{y}') == [{'y': 4}]


    def _test_user_functions(self, preql):
        preql("""
            func q1() = Person
            func q2() = q1
        """)
        res = preql("q2()()[id==me] {name}")
        assert is_eq(res, [("Erez Shinan",)])

        preql("""
            func query3() = Person[id!=me]
            func query6(c) = query3()[country==c]
            func query7() = query6
            func query8(ccc) = query7()(ccc)
        """)

        res = preql("query6(isr){name}")
        assert is_eq(res, [("Ephraim Kishon",)]), type(res)

        res = preql("query7()")
        assert isinstance(res, UserFunction)
        res = preql.query7()
        assert isinstance(res, UserFunction)

        res = preql("query8(isr) {name}")
        assert is_eq(res, [("Ephraim Kishon",)])

        preql("func languages() = Country{language}")
        # res = preql("distinct(languages())")
        # assert is_eq(res, [("he",), ("en",)])

    def test_user_functions2(self):
        p = self.Preql()
        p("""
            func f(x: int, y:list[string]) = 0
        """)
        assert p('type(f)') == T.function[T.int, T.list[T.string]]

    def _test_joins(self, preql):
        nonenglish_speakers = [
            ("Ephraim Kishon", None),
            ("Erez Shinan", None),
        ]
        english_speakers = [
            ("Eric Blaire", "England"),
            ("H.G. Wells", "England"),
            ("John Steinbeck", "United States"),
        ]

        preql("""func manual_join() = join(c: Country[language=="en"].id, p: Person.country) { p.name, country: c.name } order {name}""")
        res = preql.manual_join()
        assert is_eq(res, english_speakers)

        # Auto join
        res = preql(""" join(c: Country[language=="en"], p: Person) { p.name, country: c.name } order {name}""")
        assert is_eq(res, english_speakers), res

        res = preql(""" join(p: Person, c: Country[language=="en"]) { p.name, country: c.name } order {name}""")
        assert is_eq(res, english_speakers)

        # Left joins
        res = preql(""" leftjoin(p:Person.country, c: Country[language=="en"].id) { p.name, country: c.name } order {name}""")
        assert is_eq(res, nonenglish_speakers + english_speakers)

        res = preql(""" leftjoin(c: Country[language=="en"].id, p:Person.country) { p.name, country: c.name } order {name}""")
        assert is_eq(res, english_speakers)

        # Auto left joins

        res = preql(""" leftjoin(p:Person, c: Country[language=="en"]) { p.name, country: c.name } order {name}""")
        assert is_eq(res, nonenglish_speakers + english_speakers)

        res = preql(""" leftjoin(c: Country[language=="en"], p:Person) { p.name, country: c.name } order {name}""")
        assert is_eq(res, english_speakers)

        res = preql(""" leftjoin(c: Country, p:Person[id==me]) { person: p.name, country: c.name } order {country}""")
        expected = [
            (None, "England"),
            ("Erez Shinan", "Israel"),
            (None, "United States"),
        ]
        assert is_eq(res, expected)

        preql("""func j() = join(c: Country[language=="en"], p: Person)""")
        res = preql("j() {person: p.name, country: c.name} order {person}")
        assert is_eq(res, english_speakers)

        # res = preql("""count(joinall(a: [1,2,3], b: ["a", "b", "c"]))""")

    def test_join_to_temptable(self):
        preql = self.Preql()
        preql("""
            l1 = [1, 2, 3]
            l2 = [1, 2, 4]
            t = temptable(leftjoin(a: l1.item, b: l2.item))

            q1 = t[a.item == 1] {a.item}
            q2 = t[b.item==null] {a.item}
            #q3 = t[b==null] {a}
        """)

        assert list(preql.q1) == [{'item': 1}]
        assert list(preql.q2) == [{'item': 3}]
        # assert list(preql.q3) == [{'a': {'item': 3}}]    # TODO

    @uses_tables('Point')
    def test_update(self):
        preql = self.Preql()
        preql("""
            table Point {x: int, y: int}
            new Point(1,3)
            new Point(2,7)
            new Point(3,1)
            new Point(4,2)

            func p() = Point[x==3] update{y: y + 13}
            """)

        # TODO better syntax
        self.assertEqual( preql.p().to_json()[0]['y'], 14)
        self.assertEqual( preql.p().to_json()[0]['y'], 27)


    def test_triple_join(self):
        preql = self.Preql()
        preql(""" join(a: [1..10].item, b: [2..20].item, c:[3,5,15].item) {c.item} """) == [3, 5]



    @uses_tables('Point')
    def test_SQL(self):
        preql = self.Preql()
        preql("""
            table Point {x: int, y: int}
            new Point(1,3)
            new Point(2,7)
            new Point(3,1)
            new Point(4,2)

            x = 4
            func f1() = SQL(int, "$x+5")
            func f2() = SQL(Point, "SELECT * FROM $Point WHERE x > 2")
            func f3() = SQL(Point, "SELECT * FROM $Point") { x: x /~ 2 => y}
            zz = Point[x==2]
            func f4() = SQL(Point, "SELECT * FROM $zz") {y}

            """)

        self.assertEqual( preql.f1(), 9)
        self.assertEqual( len(preql.f2()), 2)
        self.assertEqual( len(preql.f3()), 3)
        # self.assertEqual( preql.f3().to_json()[0]['y'], [3] )     # bad test, assumes ordering
        self.assertEqual( len(preql.f4()), 1)
        self.assertEqual( preql.f4().to_json(), [{'y': 7}])

    def test_SQL2(self):
        preql = self.Preql()
        preql("""
        a = [1..10]
        func f() = SQL(int, "SELECT COUNT(*) FROM $a")
        """)
        assert preql.f() == 9

    def test_nested_projections(self):
        preql = self.Preql()

        res1 = preql("joinall(a:[1,2], b:[2, 3]) {a.item => count(b.item)}")
        res2 = preql("joinall(a:[1,2], b:[2, 3]) {a.item => count(b)}")
        self.assertEqual( res1, res2 )

        # TODO make these work, or at least throw a graceful error
        res = [{'a': {'item': 1}, 'b': [3, 4]}, {'a': {'item': 2}, 'b': [3, 4]}]
        self.assertEqual(preql("joinall(a:[1,2], b:[3, 4]) {a => b}" ), res)

        res = [{'b': 5, 'a': [1, 2]}]
        self.assertEqual(preql("joinall(a:[1,2], b:[2, 3]) {a: a.item => b: sum(b.item)} {b => a}"), res)
        # preql("joinall(a:[1,2], b:[2, 3]) {a: a.item => b: b.item} {count(b) => a}")

        res = preql("one joinall(a:[1,2], b:[2, 3]) {a: a.item => b: count(b.item)} {b => a: count(a)}")
        self.assertEqual( res, {'b':2, 'a':2} )

        res1 = preql("joinall(a:[1,2], b:[2, 3]) {b{v:item}, a}")
        res2 = preql("joinall(a:[1,2], b:[2, 3]) {b{v:item}, a{item}}")
        self.assertEqual( res1, res2 )
        res3 = preql("joinall(a:[1,2], b:[2, 3]) {b{v:item, ...}, a{...}}")
        self.assertEqual( res1, res3 )

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) ")
        assert len(res1) == 8
        res2 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab, c}")
        assert res1 == res2

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab.a, ab.b, c}")
        assert len(res1) == 8

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab.a.item, ab.b.item, c}")
        assert len(res1) == 8

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab {b: b.item, a: a.item}, c}[..1]")
        self.assertEqual(res1.to_json(), [{'ab': {'b': 2, 'a': 1}, 'c': {'item': 4}}])

    def test_nested2(self):
        preql = self.Preql()

        assert preql(" [1] {a:{b:{item}}} ") == [{'a': {'b': {'item': 1}}}]
        assert preql("[1] {a:{item}}") == preql("[1] {a:{item}} {a}")
        assert preql("[1] {item}") == preql("([1] {a:{item}}) {a.item}")


    def test_agg_funcs(self):
        preql = self.Preql()
        r = preql('[0, 2, 0, 0, 3, 4, 0] { => count_true(item), count_false(item) }')
        assert r == [{'count_true': 3, 'count_false': 4}], r

        preql("""
            func sqsum(x) = sum(x*x)
        """)
        # self.assertEqual( preql.sqsum(2), 4 )

        self.assertEqual( preql('one [2, 4]{=> sqsum(item)}')['sqsum'], 20)
        self.assertEqual( preql('sum([2, 4])'), 6)
        self.assertEqual( preql.sum([2, 4]), 6 )
        # TODO sqsum([2,4])  -->  sum([2,4]*[2,4]) --> sum([4, 16])


    def test_strings(self):
        preql = self.Preql()
        self.assertEqual( preql('upper("ba")'), "BA" )
        self.assertEqual( preql('lower("BA")'), "ba" )
        self.assertEqual( preql('"ba" in "kabab"'), True )
        self.assertEqual( preql('"ba" !in "kabab"'), False )
        self.assertEqual( preql('"bak" in "kabab"'), False )
        self.assertEqual( preql('"bak" !in "kabab"'), True )

        self.assertEqual( preql('"hello"[0..3]'), "hel" )
        self.assertEqual( preql('"hello"[1..]'), "ello" )
        self.assertEqual( preql('"hello"[..1]'), "h" )
        self.assertEqual( preql('"hello"[2..4]'), "ll" )

        self.assertEqual( preql('length("hello")'), 5 )
        self.assertEqual( preql('list(["hello"]{length(item)})'), [5] )

    def test_casts(self):
        preql = self.Preql()
        self.assertEqual( preql('type(float(1))'), T.float )
        self.assertEqual( preql('type(int(float(1)))'), T.int )
        self.assertEqual( type(preql('list[float]([1,2])')[0]), float)
        self.assertEqual( type(preql('list[int](list[float]([1,2]))')[0]), int)
        self.assertEqual( preql('list[int]([1.2, 3.4])'), [1,3])
        self.assertEqual( preql('type(list(list([1,2]{item+1}){item+1}))'), T.list[T.int])
        self.assertEqual( preql('list(list([1,2]{item+1}){item+1})'), [3,4])

        self.assertEqual( preql('list(["1", "2"]{int(item)})'), [1,2])

        try:
            self.assertEqual( preql('list(["1", "2f"]{int(item)})'), [1,2])
        except Signal:
            pass
        else:
            assert False


    def test_lists2(self):
        preql = self.Preql()
        preql('''
            func in_list(x) = [1,2,3] {item in x{item}}
            func test() = in_list([2, 3])
        ''')
        self.assertEqual( list(preql.test()), [{'_':x} for x in [0, 1, 1]])

    def test_range(self):
        preql = self.Preql()
        preql('''
            func to20() = [..20]
            func abc() = [1..3]
            func adult() = [18..]
        ''')

        assert preql.to20() == list(range(20))
        assert preql.abc() == list(range(1,3))

        try:
            assert preql('adult()[..10]') == list(range(18, 28))
        except Signal as e:
            assert e.type <= T.NotImplementedError
            assert preql._interp.state.db.target in (mysql, bigquery)   # Infinite series not supported
            return

        assert preql('adult()[..10]') == list(range(18, 28))
        assert preql('adult()[..10] + adult()[..1]') == list(range(18, 28)) + [18]
        self.assertEqual( preql('list( (adult()[..10] + adult()[..1]) {item + 1} )') , list(range(19, 29)) + [19] )


    @uses_tables('B', 'A')
    def test_rowtype(self):
        preql = self.Preql()
        preql('''
            table A { x: int }
            a = new A(4)

            table B { a: A }
            b = new B(a)

            eq1 = (a == a)
            eq2 = (b == b)
            eq3 = (a == b)

        ''')

        if preql._interp.state.db.target is not bigquery:
            # bigquery's id work differently
            self.assertEqual(preql.a, {'id': 1, 'x': 4})
        self.assertEqual(preql('a.x'), 4)
        self.assertEqual(preql('b.a.x'), 4)

        self.assertEqual(preql.eq1, True)
        self.assertEqual(preql.eq2, True)
        # self.assertEqual(preql.eq3, False)    # TODO check table type


    def test_vararg(self):
        preql = self.Preql()
        preql('''
            func f(...x) = x
        ''')

        self.assertEqual(preql('f(a:1, b:2)'), {'a':1, 'b':2})
        self.assertEqual(preql('f(a:1, b:f(c:3, d:4)).b.c'), 3)

        preql('''
            x1 = f(a:1, b:2)
            x2 = f(...x1)
        ''')

        self.assertEqual(preql.x1, preql.x2)
        # self.assertEqual(preql('x1 == x2'), True) # TODO


    @uses_tables('Node', 'Square', 'a')
    def test_methods(self):
        preql = self.Preql()
        base_attrs = dict(T.table.proto_attrs)
        preql('''
            table Square {
                size: float

                func area() = size * size
                func is_area_larger(num) = area() > num
            }

            s = new Square(4)
            size4 = Square[size==4]

            table a {
                size: int
            }
        ''')

        assert base_attrs == T.table.proto_attrs
        self.assertRaises(Signal, preql, 'a{area()}')

        # self.assertEqual( preql('s.area()'), 16 ) # TODO
        # self.assertEqual( preql('Square[size==4].area()'), 16 )
        self.assertEqual( preql('size4{ area() }').to_json(), [{'area': 16.0}] )
        self.assertEqual( preql('count(Square[area() > 18.0])'), 0 )
        self.assertEqual( preql('count(Square[area() < 18.0])'), 1 )
        self.assertEqual( preql('count(Square[is_area_larger(18.0)])'), 0 )
        self.assertEqual( preql('count(Square[is_area_larger(14.0)])'), 1 )

        # preql = self.Preql()  # TODO why does this cause errors later on?
        preql('''
            table Node {
                parent: Node?

                func children() = join(s:this.id, n:Node.parent) {n}
            }

            a = new Node(null)
            b = new Node(a)
            c = new Node(a)

        ''')
        self.assertEqual( preql('count(Node[parent==null].children())'), 2 )


    def _test_groupby(self, preql):
        assert preql('one one [1,2,3]{=>sum(item*item)}') == 14

        res = preql("Country {language => count(id)} order {language}")
        assert is_eq(res, [("en", 2), ("he", 1)]), res

        assert len(preql("Country {=> first(id)}")) == 1

        res = preql("join(p:Person, c:Country) {country:c.name => population:count(p.id)} order {country}")
        assert is_eq(res, [
            ("England", 2),
            ("Israel", 2),
            ("United States", 1),
        ]), res

        res = preql("join(p:Person order {name}, c:Country) {country:c.name => citizens: p.name} order {country}")
        for a, b in zip(res, [
            ("England", ["Eric Blaire", "H.G. Wells"]),
            ("Israel", ["Ephraim Kishon", "Erez Shinan"]),
            ("United States", ["John Steinbeck"]),
                ]):
            a = list(a.values())
            assert a[0] == b[0]
            assert set(a[1]) == set(b[1])


        res = preql("join(p:Person order {name}, c:Country) {country:c.name => citizens: p.name, count(p.id)} order {country}")
        for a, b in zip(res, [
            ("England", ["Eric Blaire", "H.G. Wells"], 2),
            ("Israel", ["Ephraim Kishon", "Erez Shinan"], 2),
            ("United States", ["John Steinbeck"], 1),
                ]):
            a = list(a.values())
            assert a[0] == b[0]
            assert set(a[1]) == set(b[1])
            assert a[2] == b[2]

        res = preql('[1,2,3]{=>sum(item*item)}')
        assert res == [{'sum': 14}], list(res)

    def test_compare(self):
        preql = self.Preql()
        assert preql('3 != "3"')
        self.assertEqual( preql("""null != 1"""), True )

        self.assertEqual( preql("""1 == 1"""), True )
        self.assertEqual( preql("""1 != 1"""), False )
        self.assertEqual( preql("""1 == 2"""), False )
        self.assertEqual( preql("""1 != 2"""), True )
        self.assertEqual( preql("""1 > 2"""), False )
        self.assertEqual( preql("""1 >= 2"""), False )
        self.assertEqual( preql("""2 >= 1"""), True )
        self.assertEqual( preql("""2 > 1"""), True )
        self.assertEqual( preql(""" "a" == "a" """), True )
        self.assertEqual( preql(""" "a" == "b" """), False )
        self.assertEqual( preql(""" "a" != "b" """), True )

        self.assertEqual( preql("""1 in [1,2,3]"""), True )
        self.assertEqual( preql("""1 !in [1,2,3]"""), False )
        self.assertEqual( preql("""4 in [1,2,3]"""), False )

        # Auto-casts. Might be disabled in the future
        # XXX yet "1" != 1  (TODO)
        self.assertEqual( preql("""'4' in [1,2,3]"""), False )
        self.assertEqual( preql("""'3' in [1,2,3]"""), True )
        self.assertEqual( preql("""4 in ['1','2','3']"""), False )
        self.assertEqual( preql("""3 in ['1','2','3']"""), True )

        # TODO
        self._assertSignal( T.TypeError, preql, """2 > "a" """)
        self._assertSignal( T.TypeError, preql, """ 1 == [2] """)
        assert preql('3 != "3"')

        self._assertSignal(T.TypeError, preql, """ [1] in [2] """)
        self._assertSignal(T.DbQueryError, preql, """ "a" in [2] """)
        self._assertSignal(T.DbQueryError, preql, """ 4 in ["a", "B"] """)
        preql.rollback()

        self.assertEqual( preql("""null == null"""), True )
        self.assertEqual( preql("""null != null"""), False )
        self.assertEqual( preql("""null != 1"""), True )
        self.assertEqual( preql(""" "a" != null """), True )



    def test_list_ops(self):
        # TODO should be consistent - always table, or always array
        preql = self.Preql()

        res = preql("""[1,2,3]""")
        assert res == [1,2,3], res
        res = preql("""[1,2,3] + [5,6]""")
        assert res == [1,2,3,5,6]

        res = preql("""[1,2,3] | [3,4]""")
        self.assertEqual(set(res), {1,2,3,4})

        # XXX not supported by mysql, but can be done using NOT IN
        try:
            res = preql("""[1,2,3] - [3,4]""")
            assert res == [1,2]
        except Signal as e:
            assert e.type <= T.NotImplementedError
            assert preql._interp.state.db.target is mysql

        res = preql("""[1,2,3]{v:item*2}[v < 5]""")
        assert res == [{'v': 2}, {'v': 4}], res

        res = preql("""[1,2,3]{v:item*2}[v in [2,6]]""")
        assert res == [{'v': 2}, {'v': 6}], res

        res = preql("""[1,2,3]{v:item*2}[v !in [2,6]]""")
        assert res == [{'v': 4}], res

        try:
            res = preql("""enum([1,8,4,4])[index+1==item]{item}""")
        except Signal as e:
            assert e.type <= T.NotImplementedError
            assert preql._interp.state.db.target is sqlite
            import sqlite3
            assert sqlite3.sqlite_version_info < (3, 25)    # Only excute
        else:
            assert res == [{'item': 1}, {'item': 4}]

        res = preql("""[1,2,3][..2]""")
        assert res == [1,2]

        res = preql("""[1,2,3][1..]""")
        assert res == [2,3]

        res = preql("""[1,2,3][1..2]""")
        assert res == [2]

        self.assertEqual( preql("""[1,2,3][1..1]"""), [])
        self.assertEqual( preql("[] {x:0}"), [])
        self._assertSignal(T.TypeError, preql, """["a", 1]""")
        self._assertSignal(T.TypeError, preql, """[1] {a: 1, a: 2} """)
        self._assertSignal(T.TypeError, preql, """[1] {a: 1 => a: 2} """)

        res ,= preql("""[1] {null, null => null, null}""")
        self.assertEqual(list(res.values()) , [None, None, None, None])

    def _test_temptable(self, preql):
        english_speakers = [
            ("Eric Blaire", "England"),
            ("H.G. Wells", "England"),
            ("John Steinbeck", "United States"),
        ]

        preql("""english_countries = temptable(Country[language=="en"], true)""")
        res = preql("english_countries{name} order {name}")
        assert is_eq(res, [("England",), ("United States",)])

        preql("""names = temptable(Person{name}) """)
        res = preql('names{name} order {name}')
        assert is_eq(res, [
            ("Ephraim Kishon",),
            ("Erez Shinan",),
            ("Eric Blaire",),
            ("H.G. Wells",),
            ("John Steinbeck",),
        ])

        # temptable join
        res = preql(""" temptable(join(c: Country[language=="en"], p: Person)) {person:p.name, country:c.name} order {person}""")
        assert is_eq(res, english_speakers)

        res = preql(""" temptable(temptable(Person, true)[name=="Erez Shinan"], true){name} """) # 2 temp tables
        assert is_eq(res, [("Erez Shinan",)])

    @uses_tables("p", "A", "B", "Person", "Country")
    def test_copy_rows(self):
        preql = self.Preql()
        preql('''
            table A { x: int }
            table B { x: int }

            new A(2)
            B += A
        ''')
        assert len(preql.B) == 1, preql.B

        preql(''' B += A ''')
        assert len(preql.B) == 2

        preql(''' B += A ''')
        assert len(preql.B) == 3

        preql(''' B += [2, 3] {x: item} ''')
        assert len(preql.B) == 5

        preql(''' A += B ''')
        assert len(preql.A) == 6

        # TODO Test, what if fields mismatch?

        preql.load('country_person.pql', rel_to=__file__)

        preql('''
            table p = Person {... !id}
            p += Person {name, country}
        ''')
        assert len(preql.p) == 2 * len(preql.Person)


    def test_one(self):
        preql = self.Preql()
        preql('''
            table A { x: int }
            table B { x: int }

            new A(2)
        ''')

        self.assertEqual( preql('one A{x}'), {'x': 2} )
        self.assertEqual( preql('one? A{x}'), {'x': 2} )
        self.assertEqual( preql('one? B'), None )

        # XXX ValueError
        self._assertSignal(T.ValueError, preql, 'one B')

        self.assertEqual( preql('one [2]'), 2 )
        self.assertEqual( preql('one? []'), None )
        self._assertSignal(T.ValueError, preql, 'one [1,2]')
        self._assertSignal(T.ValueError, preql, 'one? [1,2]')
        self._assertSignal(T.ValueError, preql, 'one []')

    def _assertSignal(self, sig_type, f, *args):
        try:
            return f(*args)
        except Signal as e:
            assert e.type <= sig_type, e
        else:
            assert False

    def _test_new(self):
        preql = self.Preql()
        preql('''
            table A { x: int }
            a = new A(1)
        ''')

        self.assertEqual( preql.a , preql('one A[x==1]') )

        # TODO
        # assert preql('a == A[x==1]')

    @uses_tables('A')
    def test_delete(self):
        preql = self.Preql()
        preql('''
            table A {
                x: int
            }
            new A(1)
            new A(2)
        ''')

        assert preql('count(A)') == 2
        res = preql('A delete [x==1]')
        assert len(res) == 1, res
        assert preql('count(A)') == 1
        res = preql('A delete [x==1]')
        assert len(res) == 1
        assert preql('count(A)') == 1
        res = preql('A delete [x==2]')
        assert len(res) == 0
        assert preql('count(A)') == 0


    @uses_tables('A')
    def test_text(self):
        preql = self.Preql()
        preql(r'''
            table A { x: text }

            a = new A("hello")
            b = new A("hello\nworld")
        ''')

        self.assertEqual( preql("one A[id==%r]{x}" % preql.a['id']), {'x': "hello"} )
        self.assertEqual( preql("one A[id==%r]{x}" % preql.b['id']), {'x': "hello\nworld"} )


    def test_nonzero(self):
        preql = self.Preql()
        preql(r'''
            func f(x) {
                if (x) {
                    return "YES"
                } else {
                    return "NO"
                }
            }

            func apply_to_list(lst) = list(lst{f(item)})
        ''')

        assert preql.f(1) == "YES"
        assert preql.f(0) == "NO"
        assert preql.f("a") == "YES"
        assert preql.f("") == "NO"

        assert preql.apply_to_list([0, 1]) == ["NO", "YES"]
        self.assertEqual( preql.apply_to_list(["", "a"]) , ["NO", "YES"] )


    @uses_tables('A')
    def test_column_default(self):
        preql = self.Preql()
        preql('''
            table A {
                x: int
                y: int = 2
            }

            a1 = new A(1)
            a2 = new A(2, 1)
        ''')

        self.assertEqual( preql('A{y} order {^y}'), [{'y': 2}, {'y': 1}] )
        assert preql('a2.y') == 1


    @uses_tables('Circle', 'Box', 'NamedLine', 'tmp1', 'tmp2')
    def test_structs(self):
        preql = self.Preql()
        preql.load('box_circle.pql', rel_to=__file__)

        res1 = preql.circles_contained1()
        res2 = preql.circles_contained2()
        res3 = preql("temptable(circles_contained2()) {...!id}")

        assert res1 == res2, (res1, res2)
        assert res2 == res3, (list(res2), list(res3))

    def test_struct_inline(self):
        preql = self.Preql()
        res = preql('list(join(a:[1..10].item, b:[8..20].item) {...a})')
        assert res == [8,9], res

        res = preql('joinall(a:[1], b:[2]) {...a, ...b}')
        assert res == [{'item': 1, 'item1': 2}]


    @uses_tables('a')
    def test_names(self):
        p = self.Preql()
        try:
            p.a
        except Signal as s:
            assert s.type <= T.NameError
            pass

        p('''
            table a {x: int}
        ''')

        self.assertEqual( p("list(names(a)[not (type ~ 'function%')]{name})"), ['id', 'x'] )
        self.assertEqual( p('columns(a)'), {'id': p.t_id, 'x': p.int} )



    @uses_tables('Person')
    def test_simple1(self):
        # TODO uncomment these tests
        preql = self.Preql()
        preql.load('simple1.pql', rel_to=__file__)

        self.assertEqual({x['name'] for x in preql.english}, {'Eric Blaire', 'H.G. Wells'})
        assert [x['name'] for x in preql.by_country('Israel')] == ['Erez Shinan']

        assert preql.english2 == [{'name': 'H.G. Wells'}, {'name': 'Eric Blaire'}]
        # assert preql.english3().json() == [{'n': 'H.G. Wells'}, {'n': 'Eric Blaire'}] # TODO

        # assert preql.person1() == [{'id': 1, 'name': 'Erez Shinan', 'country': 'Israel'}]
        # assert preql.person1b() == [{'id': 2, 'name': 'Eric Blaire', 'country': 'England'}]
        assert preql.demography == [{'country': 'England', 'population': 2}, {'country': 'Israel', 'population': 1}], list(preql.demography)

        # expected = [{'country': 'England', 'population': ['Eric Blaire', 'H.G. Wells']}, {'country': 'Israel', 'population': ['Erez Shinan']}]
        # res = preql('Person {country => population: name}')
        # assert expected == res, (res, expected)



    @skip("Not ready yet")
    def test_simple2(self):
        assert False, "Not ready yet"
        # Implicit joins
        preql = self.Preql()
        preql.load('simple2.pql', rel_to=__file__)

        res = [
            {'id': 3, 'name': 'Eric Blaire', 'country': 2},
            {'id': 4, 'name': 'H.G. Wells', 'country': 2},
            {'id': 5, 'name': 'John Steinbeck', 'country': 3}
        ]
        assert preql.english_speakers().json() == res   # TODO country should probably be a dict?

        res = [
                {'name': 'Erez Shinan', 'country.language': 'he'},
                {'name': 'Ephraim Kishon', 'country.language': 'he'},
                {'name': 'Eric Blaire', 'country.language': 'en'},
                {'name': 'H.G. Wells', 'country.language': 'en'},
                {'name': 'John Steinbeck', 'country.language': 'en'}
            ]
        assert preql.person_and_language().json() == res
        # print( preql.others_from_my_country() )   # TODO
        res = [{'language': 'en', 'name': ['England', 'United States']}, {'language': 'he', 'name': ['Israel']}]
        assert preql.country_by_language().json() == res

        res = [{'country.name': 'England', 'count_id': 2}, {'country.name': 'Israel', 'count_id': 2}, {'country.name': 'United States', 'count_id': 1}]
        assert preql.population_count().json() == res   # XXX automatic name might change

        res = [{'name': 'England', 'count_citizens': 2}, {'name': 'Israel', 'count_citizens': 2}, {'name': 'United States', 'count_citizens': 1}]
        assert preql.population_count2().json() == res  # XXX automatic name might change

        res = [{'name': 'England', 'citizens.name': ['Eric Blaire', 'H.G. Wells']},
               {'name': 'Israel', 'citizens.name': ['Ephraim Kishon', 'Erez Shinan']},
               {'name': 'United States', 'citizens.name': ['John Steinbeck']}]
        assert preql.citizens_list().json() == res

        res = [{'id': 1, 'name': 'Erez Shinan', 'country.name': 'Israel'},
               {'id': 2, 'name': 'Ephraim Kishon', 'country.name': 'Israel'},
               {'id': 3, 'name': 'Eric Blaire', 'country.name': 'England'},
               {'id': 4, 'name': 'H.G. Wells', 'country.name': 'England'},
               {'id': 5, 'name': 'John Steinbeck', 'country.name': 'United States'}]

        assert preql.person_and_country().json() == res, preql.person_and_country().json()
        # print( preql.person_and_country_join().json() )   # TODO
        # print( preql.person_and_country_freejoin().json() )

        res = [{'c': 1, 'name': ['United States']}, {'c': 2, 'name': ['England', 'Israel']}]
        assert preql('Country {name => c: count(citizens)} {c => name}').json() == res

        # Test that empty countries are still included (i.e left-join, not inner join)
        aus = preql('new Country("Australia", "mumbo-jumbo")').id
        assert preql('Country {name => c: count(citizens)} [name="Australia"]').json() == [{'name': 'Australia', 'c': 0}]
        assert preql('Country {name => names: citizens.name} [name="Australia"]').json() == [{'name': 'Australia', 'names': []}]

    @skip("Not ready yet")
    def test_m2m(self):
        assert False, "Not ready yet"
        preql = self.Preql()
        preql('''
            table A:
                item: integer

            table B:
                item: integer

            table A_B:
                a: A -> ab
                b: B -> ab

        ''')

        As = preql.add_many('A', [[x] for x in range(10)])
        Bs = preql.add_many('B', [[x] for x in range(10)])
        ABs = preql.add_many('A_B', [
            [a,b]
            for i, a in enumerate(As)
            for j, b in enumerate(Bs)
            if i*2 == j
            ])

        res = [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}, {'a': 3, 'b': 6}, {'a': 4, 'b': 8}]

        assert (preql('A_B {a: a.item, b: b.item}').json()) == res

        res = [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}, {'a': 3, 'b': 6}, {'a': 4, 'b': 8},
               {'a': 5, 'b': None}, {'a': 6, 'b': None}, {'a': 7, 'b': None}, {'a': 8, 'b': None}, {'a': 9, 'b': None}]
        assert (preql('A {a: item, b: ab.b.item}').json()) == res

        res = [{'a': 0, 'b': 0}, {'a': None, 'b': 1}, {'a': 1, 'b': 2}, {'a': None, 'b': 3}, {'a': 2, 'b': 4},
               {'a': None, 'b': 5}, {'a': 3, 'b': 6}, {'a': None, 'b': 7}, {'a': 4, 'b': 8}, {'a': None, 'b': 9}]
        assert (preql('B {a: ab.a.item, b: item}').json()) == res

        assert (preql('B [ab.a.item=2] {item}').json()) == [{'item': 4}]

        assert (preql('A_B [a.item=2] {v:b.item}').json()) == [{'v': 4}]

        res = [{'a.item': 0, 'b.item': 0}, {'a.item': 1, 'b.item': 2},
            {'a.item': 2, 'b.item': 4}, {'a.item': 3, 'b.item': 6},
            {'a.item': 4, 'b.item': 8}]
        assert (preql('A_B {a.item, b.item}').json() ) == res
        assert (preql('A_B {a, b} {a.item, b.item}').json() ) == res

    @uses_tables('a')
    def test_table_from_expr(self):
        p = self.Preql()
        p("""
            table a = [1..3]
            new a(5)
        """)
        assert {x['item'] for x in p.a} == {1,2,5}, p.a


    @uses_tables('A')
    def test_partial_table(self):
        p = self.Preql()
        p("""
            table A {
                a: int
                b: int?
                c: string
                d: float
                e: bool
                g: text
            }

            new A(1, null, "hello", 3.14, true, "world")

            A = null
            assert A == null
        """)
        assert p.A is None

        p("""
            table A {
                d: float
                c: string
                e: bool
            }

            assert count(A[c ~ "hello"]) == 1
            assert count(A[c ~ "hell"]) == 0
            assert (one one A{d}) == 3.14
        """)

    @uses_tables('A')
    def test_partial_table2(self):
        p = self.Preql()
        p("""
            table A = [1, 2, 3]

            A = null
        """)
        assert p.A is None

        p("""
            table A {...}
            new A(10)
        """)


    @skip("Not ready yet")
    def test_self_reference(self):
        # assert False, "Not ready yet"
        preql = self.Preql()
        preql('''
            table Person {
                name: string
                parent: Person? -> children
            }
        ''')

        abraham = preql('new Person("Abraham", null)')
        isaac = preql('new Person("Isaac", ab)', ab=abraham)
        jacob = preql('new Person("Jacob", isaac)', isaac=isaac)
        esau = preql('new Person("Esau", isaac)', isaac=isaac)

        assert preql('Person[name=="Jacob"] {name: parent.name}') == [{'name': 'Isaac'}]
        # assert (preql('Person[name="Jacob"] {name: parent.parent.name}').json()) == [{'name': 'Abraham'}] # TODO

        res = [{'name': 'Abraham', 'c': 1}, {'name': 'Isaac', 'c': 2}]
        assert ( preql('Person {name => c: count(children)} [c>0]').json()) == res

        res = [{'name': 'Abraham', 'children.name': ['Isaac']},
            {'name': 'Esau', 'children.name': []},
            {'name': 'Isaac', 'children.name': ['Esau', 'Jacob']},
            {'name': 'Jacob', 'children.name': []}]

        assert ( preql('Person {name => children.name}').json()) == res

        # assert ( preql('Person {name => children.name}').json()) == res

    @skip("Not ready yet")
    def test_m2m_with_self_reference(self):
        assert False, "Not ready yet"
        preql = self.Preql()
        preql('''
            table A:
                name: string
                parent: A? -> children

            table B:
                name: string
                parent: B? -> children

            table A_B:
                a: A -> ab
                b: B -> ab

        ''')

        a1 = preql('new A("a1", null)').id
        a2 = preql('new A("a2", a1)', a1=a1).id
        a3 = preql('new A("a3", a1)', a1=a1).id

        b1 = preql('new B("b1", null)').id
        b2 = preql('new B("b2", null)').id

        preql('new A_B(a1, b1)', a1=a1, b1=b1).id
        preql('new A_B(a2, b2)', a2=a2, b2=b2).id

        # TODO Table identity is messed up!
        # The rules should be as followed (at least for now):
        # * Reference always brings a new table ( so join(A,A) will joins two separate selects )
        # * get_attr should returns a new table, but cached, according to the originating table
        #   so {ab} creates a new reference, and {children.ab} creates yet another new reference
        #   so that two separate joins occur!
        # res = [{'ab.b.name': 'b1', 'children.ab.b.name': 'b2'}]
        res = [{'ab.b.name': 'b1', 'children.ab.b.name': None},
               {'ab.b.name': 'b1', 'children.ab.b.name': 'b2'},
               {'ab.b.name': 'b2', 'children.ab.b.name': None},
               {'ab.b.name': None, 'children.ab.b.name': None}]
        assert( preql('A {ab.b.name, children.ab.b.name}').json() ) == res

        assert( preql('A[ab.b.name="b1", children.ab.b.name="b2"] {name}').json() ) == [{'name': 'a1'}]
        # print( preql('A {children.ab.b.name}').json() )
        # print( preql('A[ab.b.name="b1"] {children.name}').json() ) # TODO get it to work

    # def test_m2m_syntax(self):
    #     preql = _Preql()
    #     preql.exec('''
    #         table A:
    #             name: string
    #             parent: A? -> children

    #         table B:
    #             name: string
    #             parent: B? -> children
    #             a_set: [A] -> b_set

    #     ''')

    #     a1 = preql('new A("a1", null)').row_id
    #     a2 = preql('new A("a2", a1)', a1=a1).row_id
    #     a3 = preql('new A("a3", a1)', a1=a1).row_id

    #     b1 = preql('new B("b1", null)').row_id
    #     b2 = preql('new B("b2", null)').row_id


        # print( preql('new B:a_set', b1=b1) )

    @uses_tables('A')
    def test_import_table(self):
        preql = self.Preql()

        if self.uri == BIGQUERY_URI:
            preql("""
                table A {
                    a: int
                    b: int?
                    c: string
                    d: float
                    e: bool
                    f: timestamp
                }
            """)
        else:
            preql("""
                table A {
                    a: int
                    b: int?
                    c: string
                    d: float
                    e: bool
                    f: timestamp
                    g: text
                }
            """)

        a_type = preql('type(A{...!id})')   # TODO convert 'id' to t_id
        preql._reset_interpreter()
        self.assertRaises(Signal, preql, 'A')

        preql("""
            table A {...}
        """)
        t = preql('type(A{...!id})')
        assert a_type == t, (a_type, t)

        preql._reset_interpreter()
        preql("""
            table A {c: int, ...}
        """)
        t = preql('type(A{...!id})')
        assert a_type != t

        preql._reset_interpreter()
        preql("""
            table A {c: string, ...}
        """)
        t = preql('type(A{...!id})')
        assert a_type != t  # different order
        assert a_type.elems == t.elems

    def test_repeat(self):
        p = self.Preql()
        res = p('[1,2,3]{repeat("a", item)}')
        assert res == [{'repeat': "a"*i} for i in range(1,4)], res

    def test_vectorized_logic(self):
        p = self.Preql()
        p("""
        func sign(x) {
            if (x == 0) {
                return 0
            } else if (x > 0) {
                return 1
            } else {
                return -1
            }
        }

        """)
        assert p.sign(-1) == -1
        assert p.sign(0) == 0
        assert p.sign(1) == 1

        res = p('list([-2..3]{sign(item)})')
        assert res == [-1, -1, 0, 1, 1], res

        res = p('[-2..3]{=>sign(item)}')
        assert res == [{'sign': [-1, -1, 0, 1, 1]}], res

    def test_builtins(self):
        p = self.Preql()

        assert p("list([1.1, 2.3]{round(item)})") == [1.0, 2.0]
        assert p('round(1.3)') == 1.0

        assert p('list(["A", "Ab"]{length(item)})') == [1, 2]
        assert p('list(["A", "Ab"]{lower(item)})') == ["a", "ab"]
        assert p('list(["A", "Ab"]{upper(item)})') == ["A", "AB"]
        assert p('length("Ab")') == 2
        assert p('lower("Ab")') == "ab"
        assert p('upper("Ab")') == "AB"

        assert p('list(["Ab", "Aab"]{str_index("b", item)})') == [1, 2]
        assert p('str_index("b", "Ab")') == 1

        assert p('char(65)') == 'A'
        assert p('char_ord("A")') == 65
        assert p('char_range("a", "c")') == ['a', 'b', 'c']

    def test_json(self):
        p = self.Preql()
        res = p('list([1,7,3,4]{item%2 => item}{count(item)} order {count})')
        assert res == [1, 3], res

    def test_table_def_dicts(self):
        p = self.Preql()
        res = p('''[
            {a: 1, b: 2}
            {a: 10, b:20}
        ]{c: a+b}''')
        assert res == [{'c': 3}, {'c': 30}]


    def test_join_on(self):
        if self.uri == BIGQUERY_URI:
            raise SkipTest("Not supported in BigQuery")

        p = self.Preql()
        p("""
            A = [1, 3]
            B = [1, 2]
            res = leftjoin(a: A, b: B, $on: a.item > b.item)
        """)

        assert p.res == [
            {'a': {'item': 1}, 'b': {'item': None}},
            {'a': {'item': 3}, 'b': {'item': 1}},
            {'a': {'item': 3}, 'b': {'item': 2}}
        ]

    def test_dates(self):
        p = self.Preql()
        p("""
            table A {
                dt: timestamp = now()
            }

            x = new A()
            y = new A(x.dt)
            z = new A("2021-04-02 22:28:41")

        """)

        assert p.x['dt'] == p.y['dt'] != p.z['dt']

    def test_keywords(self):
        # TODO only need to run this once
        p = self.Preql()
        p('func f(a, b=4) = a + b')

        assert p.f(b=2, a=3) == 5
        assert p.f(3, b=10) == 13
        self.assertRaises(Signal, p, 'f(3, a:10)')
        self.assertRaises(Signal, p.f, p.f, 3, a=10)

    def test_threading(self):
        p = self.Preql()
        p('''
        table a = [0]

        func add_n(n) = new a(n)
        ''')

        with ThreadPool(processes=10) as pool:
            pool.map(p.add_n, range(100))

        assert len(p.a) == 101, len(p.a)
        if p._interp.state.db.target != mysql:   # Not supported
            assert p('a{item} - [..100]') == []



class TestFlow(PreqlTests):
    def test_new_freezes_values(self):
        # Test for issue #7
        p = self.Preql()
        p('''
        table a{x: float}
        row = new a(random())
        assert row.x in a{x}
        ''')

class TestTypes(PreqlTests):
    def test_types(self):
        assert T.int == T.int
        assert T.int != T.number
        assert T.int <= T.number
        assert T.int <= T.union[T.number, T.string]
        assert T.union[T.int, T.string] != T.union[T.bool, T.text]

        assert T.struct(dict(n=T.int)) == T.struct(dict(n=T.int))
        assert T.struct(dict(n=T.int)) != T.struct(dict(m=T.int))
        assert T.struct(dict(n=T.int)) != T.struct(dict(n=T.string))

        assert T.list[T.number] <= T.list
        assert T.list[T.any] <= T.list

        assert T.nulltype <= T.int.as_nullable()
        assert T.int.type <= T.type
        assert T.table(x=T.int, y=T.string).type <= T.type

        u = T.union[T.int, T.string]
        assert T.int <= u
        assert T.string <= u
        assert not (u <= T.int)
        assert u <= u

class TestFunctions(PreqlTests):
    def test_fmt(self):
         p = self.Preql()
         p("""
            a = "hello"
            b = "world"

            f1 = fmt("")
            f2 = fmt("a")
            f3 = fmt("a b c $a")
            f4 = fmt("a b c $a $b!")
            f5 = fmt("$a my $b!")
         """)

         assert p.f1 == ''
         assert p.f2 == 'a'
         assert p.f3 == 'a b c hello'
         assert p.f4 == 'a b c hello world!'
         assert p.f5 == 'hello my world!'


@parameterized_class(("name", "uri"), NORMAL_TARGETS)
class TestStdlib(PreqlTests):
    def test_round(self):
        p = self.Preql()
        n = 1928.9182
        assert p.round(n) == p.round(n, 0) == 1929
        assert float(p.round(n, 1)) == 1928.9
        assert float(p.round(n, 2)) == 1928.92
        assert float(p.round(n, -1)) == 1930
        assert float(p.round(n, -3)) == 2000
        assert float(p.round(-59.9)) == -60
        assert float(p.round(-4.535,2)) == -4.54



class TestPandas(PreqlTests):
    def test_pandas(self):
        from pandas import DataFrame
        f = DataFrame([[1,2,"a"], [4,5,"b"], [7,8,"c"]], columns=['x', 'y', 'z'])
        p = self.Preql()
        p.import_pandas(x=f)
        assert (p('x{... !id}').to_pandas() == f).all().all()