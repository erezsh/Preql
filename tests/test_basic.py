from unittest import skip

from parameterized import parameterized_class

from preql import Preql
from preql.pql_objects import UserFunction
from preql.exceptions import PreqlError, pql_TypeError, pql_SyntaxError, pql_ValueError, pql_NameNotFound
from preql.interp_common import pql_TypeError
from preql import sql, settings
from preql.pql_types import T

from .common import PreqlTests, SQLITE_URI, POSTGRES_URI, MYSQL_URI



def is_eq(a, b):
    a = [tuple(row.values()) for row in a]
    return a == b

@parameterized_class(("name", "uri", "optimized"), [
    ("Normal_Lt", SQLITE_URI, True),
    ("Normal_Pg", POSTGRES_URI, True),
    ("Normal_My", MYSQL_URI, True),
    ("Unoptimized_Lt", SQLITE_URI, False),
    ("Unoptimized_Pg", POSTGRES_URI, False),
])
class BasicTests(PreqlTests):
    def Preql(self, **kw):
        settings.optimize = self.optimized
        preql = Preql(self.uri, **kw)
        self.preql = preql
        return preql

    def setUp(self):
        self.preql = None

    def tearDown(self):
        if self.preql:
            self.preql.engine.rollback()

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
        self.assertEqual( preql("[1,2,3]{v: value/~2 => sum(value)}").to_json(), [{'v':0, 'sum': 1}, {'v':1, 'sum':5}])
        self.assertEqual( preql("[1,2,3]{value: value/~2 => sum(value)}").to_json(), [{'value':0, 'sum': 1}, {'value':1, 'sum':5}])


        preql("""func query1() = Country[language=="en"]{name}""")

        assert is_eq(preql.query1(), [("England",), ("United States",)]), preql.query1()

        assert is_eq(preql("Person[country==isr]{name}"), [("Erez Shinan",), ("Ephraim Kishon",)])

        res = preql("Person[id!=me]{name}")
        assert is_eq(res, [("Ephraim Kishon",), ("Eric Blaire",), ("H.G. Wells",), ("John Steinbeck",)])

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

        self.assertRaises( pql_SyntaxError, preql, 'Person {x: ...}')
        self.assertRaises( pql_SyntaxError, preql, 'Person {...+"a", 2}')

    def _test_ellipsis_exclude(self, preql):
        self.assertEqual( preql('Person {name, ... !id !country}[name=="Erez Shinan"]'), [{'name': 'Erez Shinan'}] )

        assert list(preql('Person {name, ... !id !country}')[0].keys()) == ['name']
        assert list(preql('Person {country, ... !name}')[0].keys()) == ['country', 'id']
        assert list(preql('Person {... !name, id}')[0].keys()) == ['country', 'id']
        assert list(preql('Person {country, ... !name, id}')[0].keys()) == ['country', 'id']


        self.assertRaises(pql_NameNotFound, preql, '[3]{... !hello}')
        self.assertRaises(PreqlError, preql, '[3]{... !value}')

        # TODO exception when name doesn't exist

    def test_arith(self):
        preql = self.Preql()
        assert preql("1 + 2 / 4") == 1.5
        assert preql("1 + 2 /~ 4 + 1") == 2
        assert preql('"a" + "b"') == "ab"
        assert preql('"a" * 3') == "aaa"
        assert preql('"ab" * 3') == "ababab"
        assert preql('"a" + "b"*2 + "c"') == 'abbc'
        assert preql('"a" ~ "a%"')
        assert preql('"abc" ~ "a%"')
        assert preql('"abc" ~ "a%c"')
        assert not preql('"ab" ~ "a%c"')

        self.assertRaises(pql_TypeError, preql, '"a" + 3')
        self.assertRaises(pql_TypeError, preql, '"a" ~ 3')
        self.assertRaises(pql_TypeError, preql, '"a" - "b"')
        self.assertRaises(pql_TypeError, preql, '"a" % "b"')
        self.assertRaises(pql_TypeError, preql, '3 ~ 3')


    def test_update_basic(self):
        preql = self.Preql()
        preql("""
        table Point {x: int, y: int}

        new Point(1,3)
        new Point(2,7)
        new Point(3,1)
        new Point(4,2)

        const table backup = Point

        func p() = Point[x==3] update{y: y + 13}
        """)
        assert preql.p() == [{'id': 3, 'x': 3, 'y': 14}]
        assert preql.p() == [{'id': 3, 'x': 3, 'y': 27}]
        assert preql('backup[x==3]{y}') == [{'y': 1}]
        res = preql('backup[x==3] update {y: x+y}')
        assert res == [{'id': 3, 'x': 3, 'y': 4}], res
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

    def _test_joins(self, preql):
        nonenglish_speakers = [
            ("Erez Shinan", None),
            ("Ephraim Kishon", None),
        ]
        english_speakers = [
            ("Eric Blaire", "England"),
            ("H.G. Wells", "England"),
            ("John Steinbeck", "United States"),
        ]

        preql("""func manual_join() = join(c: Country[language=="en"].id, p: Person.country) { p.name, country: c.name }""")
        res = preql.manual_join()
        assert is_eq(res, english_speakers)

        # Auto join
        res = preql(""" join(c: Country[language=="en"], p: Person) { p.name, country: c.name } """)
        assert is_eq(res, english_speakers)

        res = preql(""" join(p: Person, c: Country[language=="en"]) { p.name, country: c.name } """)
        assert is_eq(res, english_speakers)

        # Left joins
        res = preql(""" leftjoin(p:Person.country, c: Country[language=="en"].id) { p.name, country: c.name } """)
        assert is_eq(res, nonenglish_speakers + english_speakers)

        res = preql(""" leftjoin(c: Country[language=="en"].id, p:Person.country) { p.name, country: c.name } """)
        assert is_eq(res, english_speakers)

        # Auto left joins

        res = preql(""" leftjoin(p:Person, c: Country[language=="en"]) { p.name, country: c.name } """)
        assert is_eq(res, nonenglish_speakers + english_speakers)

        res = preql(""" leftjoin(c: Country[language=="en"], p:Person) { p.name, country: c.name } """)
        assert is_eq(res, english_speakers)

        res = preql(""" leftjoin(c: Country, p:Person[id==me]) { person: p.name, country: c.name } """)
        expected = [
            ("Erez Shinan", "Israel"),
            (None, "England"),
            (None, "United States"),
        ]
        assert is_eq(res, expected)

        preql("""func j() = join(c: Country[language=="en"], p: Person)""")
        res = preql("j() {person: p.name, country: c.name}")
        assert is_eq(res, english_speakers)

        # res = preql("""count(joinall(a: [1,2,3], b: ["a", "b", "c"]))""")

    def test_join_to_temptable(self):
        preql = self.Preql()
        preql("""
            l1 = [1, 2, 3]
            l2 = [1, 2, 4]
            t = temptable(leftjoin(a: l1.value, b: l2.value))

            q1 = t[a.value == 1] {a.value}
            q2 = t[b.value==null] {a.value}
            #q3 = t[b==null] {a}
        """)

        assert list(preql.q1) == [{'value': 1}]
        assert list(preql.q2) == [{'value': 3}]
        # assert list(preql.q3) == [{'a': {'value': 3}}]    # TODO

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
        if preql.engine.target == sql.sqlite:
            self.assertEqual( preql.f3().to_json()[0]['y'], '3' )
        else:
            self.assertEqual( preql.f3().to_json()[0]['y'], [3] )
        self.assertEqual( len(preql.f4()), 1)
        self.assertEqual( preql.f4().to_json(), [{'y': 7}])

    def test_nested_projections(self):
        preql = self.Preql()

        res1 = preql("joinall(a:[1,2], b:[2, 3]) {a.value => count(b.value)}")
        res2 = preql("joinall(a:[1,2], b:[2, 3]) {a.value => count(b)}")
        self.assertEqual( res1, res2 )

        # TODO make these work, or at least throw a graceful error
        res = [{'a': {'value': 1}, 'b': [3, 4]}, {'a': {'value': 2}, 'b': [3, 4]}]
        self.assertEqual(preql("joinall(a:[1,2], b:[3, 4]) {a => b}" ), res)

        res = [{'b': 5, 'a': [1, 2]}]
        self.assertEqual(preql("joinall(a:[1,2], b:[2, 3]) {a: a.value => b: sum(b.value)} {b => a}"), res)
        # preql("joinall(a:[1,2], b:[2, 3]) {a: a.value => b: b.value} {count(b) => a}")

        res = preql("one joinall(a:[1,2], b:[2, 3]) {a: a.value => b: count(b.value)} {b => a: count(a)}")
        self.assertEqual( res, {'b':2, 'a':2} )

        res1 = preql("joinall(a:[1,2], b:[2, 3]) {b{v:value}, a}")
        res2 = preql("joinall(a:[1,2], b:[2, 3]) {b{v:value}, a{value}}")
        self.assertEqual( res1, res2 )
        res3 = preql("joinall(a:[1,2], b:[2, 3]) {b{v:value, ...}, a{...}}")
        self.assertEqual( res1, res3 )

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) ")
        assert len(res1) == 8
        res2 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab, c}")
        assert res1 == res2

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab.a, ab.b, c}")
        assert len(res1) == 8

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab.a.value, ab.b.value, c}")
        assert len(res1) == 8

        res1 = preql("joinall(ab: joinall(a:[1,2], b:[2,3]), c: [4,5]) {ab {b: b.value, a: a.value}, c}[..1]")
        self.assertEqual(res1.to_json(), [{'ab': {'b': 2, 'a': 1}, 'c': {'value': 4}}])


    def test_agg_funcs(self):
        preql = self.Preql()
        r = preql('[0, 2, 0, 0, 3, 4, 0] { => count_true(value), count_false(value) }')
        assert r == [{'count_true': 3, 'count_false': 4}], r

        preql("""
            func sqsum(x) = sum(x*x)
        """)
        # self.assertEqual( preql.sqsum(2), 4 )

        self.assertEqual( preql('one [2, 4]{=> sqsum(value)}')['sqsum'], 20)
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

    def test_casts(self):
        preql = self.Preql()
        self.assertEqual( preql('type(float(1))'), T.float )
        self.assertEqual( preql('type(int(float(1)))'), T.int )
        self.assertEqual( type(preql('list[float]([1,2])')[0]), float)
        self.assertEqual( type(preql('list[int](list[float]([1,2]))')[0]), int)
        self.assertEqual( preql('list[int]([1.2, 3.4])'), [1,3])
        self.assertEqual( preql('type(list(list([1,2]{value+1}){value+1}))'), T.list[T.int])
        self.assertEqual( preql('list(list([1,2]{value+1}){value+1})'), [3,4])

    def test_lists2(self):
        preql = self.Preql()
        preql('''
            func in_list(x) = [1,2,3] {value in x{value}}
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
        assert preql('adult()[..10]') == list(range(18, 28))
        assert preql('adult()[..10] + adult()[..1]') == list(range(18, 28)) + [18]
        self.assertEqual( preql('list( (adult()[..10] + adult()[..1]) {value + 1} )') , list(range(19, 29)) + [19] )


    def test_rowtype(self):
        preql = Preql()
        preql('''
            table A { x: int }
            a = new A(4)

            table B { a: A }
            b = new B(a)

            eq1 = (a == a)
            eq2 = (b == b)
            eq3 = (a == b)

        ''')

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


    def test_methods(self):
        preql = self.Preql()
        assert not T.table.methods
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

        assert not T.table.methods
        self.assertRaises(pql_NameNotFound, preql, 'a{area()}')

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
        assert preql('one one [1,2,3]{=>sum(value*value)}') == 14

        res = preql("Country {language => count(id)} order {language}")
        assert is_eq(res, [("en", 2), ("he", 1)]), res

        assert len(preql("Country {=> first(id)}")) == 1

        res = preql("join(p:Person, c:Country) {country:c.name => population:count(p.id)} order {country}")
        assert is_eq(res, [
            ("England", 2),
            ("Israel", 2),
            ("United States", 1),
        ])

        res = preql("join(p:Person, c:Country) {country:c.name => citizens: p.name}")
        assert is_eq(res, [
            ("England", ["Eric Blaire", "H.G. Wells"]),
            ("Israel", ["Erez Shinan", "Ephraim Kishon"]),
            ("United States", ["John Steinbeck"]),
        ]), list(res)

        res = preql("join(p:Person, c:Country) {country:c.name => citizens: p.name, count(p.id)}")
        assert is_eq(res, [
            ("England", ["Eric Blaire", "H.G. Wells"], 2),
            ("Israel", ["Erez Shinan", "Ephraim Kishon"], 2),
            ("United States", ["John Steinbeck"], 1),
        ])

        res = preql('[1,2,3]{=>sum(value*value)}')
        assert res == [{'sum': 14}], list(res)

    def test_compare(self):
        preql = self.Preql()

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

        # self.assertRaises( pql_TypeError, preql, """2 > "a" """)
        # self.assertRaises( pql_TypeError, preql, """2 == "a" """)
        # self.assertRaises( pql_TypeError, preql, """ 1 == [2] """)
        self.assertRaises( pql_TypeError, preql, """ [1] in [2] """)
        self.assertRaises( pql_TypeError, preql, """ "a" in [2] """)
        # self.assertRaises( pql_TypeError, preql, """ 4 in ["a", "B"] """) # TODO good or bad?

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

        res = preql("""[1,2,3] - [3,4]""")
        assert res == [1,2]

        res = preql("""[1,2,3]{v:value*2}[v < 5]""")
        assert res == [{'v': 2}, {'v': 4}], res

        res = preql("""[1,2,3]{v:value*2}[v in [2,6]]""")
        assert res == [{'v': 2}, {'v': 6}], res

        res = preql("""[1,2,3]{v:value*2}[v !in [2,6]]""")
        assert res == [{'v': 4}], res

        res = preql("""enum([1,8,4,4])[index==value]{value}""")
        assert res == [{'value': 1}, {'value': 4}]

        res = preql("""[1,2,3][..2]""")
        assert res == [1,2]

        res = preql("""[1,2,3][1..]""")
        assert res == [2,3]

        res = preql("""[1,2,3][1..2]""")
        assert res == [2]

        self.assertEqual( preql("""[1,2,3][1..1]"""), [])
        self.assertEqual( preql("[] {x:0}"), [])
        self.assertRaises( pql_TypeError, preql, """["a", 1]""")
        self.assertRaises( pql_TypeError, preql, """[1] {a: 1, a: 2} """)
        self.assertRaises( pql_TypeError, preql, """[1] {a: 1 => a: 2} """)

        res ,= preql("""[1] {null, null => null, null}""")
        self.assertEqual(list(res.values()) , [None, None, None, None])

    def _test_temptable(self, preql):
        english_speakers = [
            ("Eric Blaire", "England"),
            ("H.G. Wells", "England"),
            ("John Steinbeck", "United States"),
        ]

        preql("""english_countries = temptable(Country[language=="en"], true)""")
        res = preql("english_countries{name}")
        assert is_eq(res, [("England",), ("United States",)])

        preql("""names = temptable(Person{name}) """)
        res = preql('names{name}')
        assert is_eq(res, [
            ("Erez Shinan",),
            ("Ephraim Kishon",),
            ("Eric Blaire",),
            ("H.G. Wells",),
            ("John Steinbeck",),
        ])

        # temptable join
        res = preql(""" temptable(join(c: Country[language=="en"], p: Person)) {person:p.name, country:c.name} """)
        assert is_eq(res, english_speakers)

        res = preql(""" temptable(temptable(Person, true)[name=="Erez Shinan"], true){name} """) # 2 temp tables
        assert is_eq(res, [("Erez Shinan",)])

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

        preql(''' B += [2, 3] {x: value} ''')
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
        self.assertRaises(pql_ValueError, preql, 'one B')

        self.assertEqual( preql('one [2]'), 2 )
        self.assertEqual( preql('one? []'), None )
        self.assertRaises(pql_ValueError, preql, 'one [1,2]')
        self.assertRaises(pql_ValueError, preql, 'one? [1,2]')
        self.assertRaises(pql_ValueError, preql, 'one []')

    def _test_new(self):
        preql = self.Preql()
        preql('''
            table A { x: int }
            a = new A(1)
        ''')

        self.assertEqual( preql.a , preql('one A[x==1]') )

        # TODO
        # assert preql('a == A[x==1]')

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

    def test_text(self):
        preql = self.Preql()
        preql(r'''
            table A { x: text }

            new A("hello")
            new A("hello\nworld")
        ''')

        self.assertEqual( preql("one A[id==1]{x}"), {'x': "hello"} )
        self.assertEqual( preql("one A[id==2]{x}"), {'x': "hello\nworld"} )


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

        assert preql('A{y}') == [{'y': 2}, {'y': 1}]
        assert preql('a2.y') == 1

    def test_structs(self):
        preql = self.Preql()
        preql.load('box_circle.pql', rel_to=__file__)

        res1 = preql.circles_contained1()
        res2 = preql.circles_contained2()
        res3 = preql("temptable(circles_contained2()) {...!id}")

        assert res1 == res2, (res1, res2)
        assert res2 == res3, (list(res2), list(res3))


    def test_names(self):
        p = self.Preql()
        try:
            p.a
        except PreqlError:
            pass

        p('''
            table a {x: int}
        ''')

        self.assertEqual( p('list(names(a){name})'), ['id', 'x'] )
        self.assertEqual( p('columns(a)'), {'id': p.t_id, 'x': p.int} )


    def test_simple1(self):
        # TODO uncomment these tests
        preql = self.Preql()
        preql.load('simple1.pql', rel_to=__file__)

        self.assertEqual([x['name'] for x in preql.english], ['Eric Blaire', 'H.G. Wells'])
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
                value: integer

            table B:
                value: integer

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

        assert (preql('A_B {a: a.value, b: b.value}').json()) == res

        res = [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}, {'a': 3, 'b': 6}, {'a': 4, 'b': 8},
               {'a': 5, 'b': None}, {'a': 6, 'b': None}, {'a': 7, 'b': None}, {'a': 8, 'b': None}, {'a': 9, 'b': None}]
        assert (preql('A {a: value, b: ab.b.value}').json()) == res

        res = [{'a': 0, 'b': 0}, {'a': None, 'b': 1}, {'a': 1, 'b': 2}, {'a': None, 'b': 3}, {'a': 2, 'b': 4},
               {'a': None, 'b': 5}, {'a': 3, 'b': 6}, {'a': None, 'b': 7}, {'a': 4, 'b': 8}, {'a': None, 'b': 9}]
        assert (preql('B {a: ab.a.value, b: value}').json()) == res

        assert (preql('B [ab.a.value=2] {value}').json()) == [{'value': 4}]

        assert (preql('A_B [a.value=2] {v:b.value}').json()) == [{'v': 4}]

        res = [{'a.value': 0, 'b.value': 0}, {'a.value': 1, 'b.value': 2},
            {'a.value': 2, 'b.value': 4}, {'a.value': 3, 'b.value': 6},
            {'a.value': 4, 'b.value': 8}]
        assert (preql('A_B {a.value, b.value}').json() ) == res
        assert (preql('A_B {a, b} {a.value, b.value}').json() ) == res


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

    def test_import_table(self):
        preql = self.Preql()
        preql("""
            table A {
                a: int
                b: int?
                c: string
                d: float
                e: bool
                f: datetime
                g: text
            }
        """)
        a_type = preql('type(A{...!id})')   # TODO convert 'id' to t_id
        preql._reset_interpreter()
        self.assertRaises(pql_NameNotFound, preql, 'A')

        preql("""
            table A {...}
        """)
        t = preql('type(A{...!id})')
        assert a_type == t

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


from pandas import DataFrame
class TestPandas(PreqlTests):
    def test_pandas(self):
        f = DataFrame([[1,2,"a"], [4,5,"b"], [7,8,"c"]], columns=['x', 'y', 'z'])
        p = self.Preql()
        p.import_pandas(x=f)
        assert (p('x{... !id}').to_pandas() == f).all().all()