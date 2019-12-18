from unittest import TestCase, skip
from copy import copy

from parameterized import parameterized_class

from preql import Preql
from preql.pql_objects import UserFunction
from preql.exceptions import PreqlError, pql_TypeError, pql_SyntaxError
from preql.interp_common import GlobalSettings, pql_TypeError


def is_eq(a, b):
    a = [tuple(row.values()) for row in a]
    return a == b

@parameterized_class(("name", "optimized"), [
    ("Normal", True),
    ("Unoptimized", False)
])
class BasicTests(TestCase):
    def Preql(self):
        GlobalSettings.Optimize = self.optimized
        return Preql()

    def test_basic1(self):
        preql = self.Preql()
        preql.load('country_person.pql', rel_to=__file__)

        self._test_basic(preql)
        self._test_ellipsis(preql)
        self._test_user_functions(preql)
        self._test_joins(preql)
        self._test_groupby(preql)
        self._test_table_ops(preql)
        self._test_temptable(preql)

    def _test_basic(self, preql):
        self.assertEqual(preql("3.14"), 3.14)

        assert preql("1") == 1
        assert preql("1 / 2") == 0.5
        assert preql("10 // 3") == 3


        preql.exec("""func query1() = Country[language=="en"]{name}""")

        assert is_eq(preql.query1(), [("England",), ("United States",)])

        assert is_eq(preql("Person[country==isr]{name}"), [("Erez Shinan",), ("Ephraim Kishon",)])

        res = preql("Person[id!=me]{name}")
        assert is_eq(res, [("Ephraim Kishon",), ("Eric Blaire",), ("H.G. Wells",), ("John Steinbeck",)])

    def _test_ellipsis(self, preql):
        assert preql('Person {name, ...}[name=="Erez Shinan"]') == [{'name': 'Erez Shinan', 'id': 1, 'country': 1}]

        assert list(preql('Person {name, ...}')[0].keys()) == ['name', 'id', 'country']
        assert list(preql('Person {country, ...}')[0].keys()) == ['country', 'id', 'name']
        assert list(preql('Person {..., id}')[0].keys()) == ['name', 'country', 'id']
        assert list(preql('Person {country, ..., id}')[0].keys()) == ['country', 'name', 'id']

        self.assertEqual( list(preql('Person {name2: name, ...}')[0].keys()), ['name2', 'id', 'country'])
        assert list(preql('Person {name2: name, ..., name3: name}')[0].keys()) == ['name2', 'id', 'country', 'name3']
        assert list(preql('Person {name: name, ...}')[0].keys()) == ['name', 'id', 'country']

        self.assertEqual( list(preql('Person {name2: name+"!", ...}')[0].keys()), ['name2', 'id', 'name', 'country'])
        self.assertEqual( list(preql('Person {name2: name+"!", ..., name3: name+"!"}')[0].keys()), ['name2', 'id', 'name', 'country', 'name3'])
        self.assertEqual( list(preql('Person {name2: name+"!", ..., name3: name}')[0].keys()), ['name2', 'id', 'country', 'name3'])

        self.assertRaises( pql_SyntaxError, preql, 'Person {x: ...}')
        self.assertRaises( pql_SyntaxError, preql, 'Person {...+"a", 2}')

    def test_arith(self):
        preql = self.Preql()
        assert preql("1 + 2 / 4") == 1.5
        assert preql("1 + 2 // 4 + 1") == 2
        assert preql('"a" + "b"') == "ab"
        assert preql('"a" * 3') == "aaa" == preql('3 * "a"')
        assert preql('"ab" * 3') == "ababab" == preql('3 * "ab"')
        assert preql('"a" + "b"*2 + "c"') == 'abbc'
        self.assertRaises(pql_TypeError, preql, '"a" + 3')


    def test_update_basic(self):
        preql = self.Preql()
        preql("""
        table Point {x: int, y: int}

        new Point(1,3)
        new Point(2,7)
        new Point(3,1)
        new Point(4,2)

        backup = temptable(Point)

        func p() = Point[x==3] update{y: y + 13}
        """)
        assert preql.p() == [{'id': 3, 'x': 3, 'y': 14}]
        assert preql.p() == [{'id': 3, 'x': 3, 'y': 27}]
        assert preql('backup[x==3]{y}') == [{'y': 1}]
        res = preql('backup[x==3] update {y: x+y}')
        assert res == [{'id': 3, 'x': 3, 'y': 4}], res
        assert preql('backup[x==3]{y}') == [{'y': 4}]

    def _test_user_functions(self, preql):
        preql.exec("""
            func q1() = Person
            func q2() = q1
        """)
        res = preql("q2()()[id==me] {name}")
        assert is_eq(res, [("Erez Shinan",)])

        preql.exec("""
            func query3() = Person[id!=me]
            func query6(c) = query3()[country==c]
            func query7() = query6
            func query8(ccc) = query7()(ccc)
        """)

        res = preql("query6(isr){name}")
        assert is_eq(res, [("Ephraim Kishon",)])

        res = preql("query7()")
        assert isinstance(res, UserFunction)
        res = preql.query7()
        assert isinstance(res, UserFunction)

        res = preql("query8(isr) {name}")
        assert is_eq(res, [("Ephraim Kishon",)])

        preql.exec("func languages() = Country{language}")
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

        preql.exec("""func manual_join() = join(c: Country[language=="en"].id, p: Person.country) { p.name, country: c.name }""")
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

        preql.exec("""func j() = join(c: Country[language=="en"], p: Person)""")
        res = preql("j() {person: p.name, country: c.name}")
        assert is_eq(res, english_speakers)


    def _test_groupby(self, preql):
        res = preql("Country {language => count(id)}")
        assert is_eq(res, [("en", 2), ("he", 1)])

        res = preql("join(p:Person, c:Country) {country:c.name => population:count(p.id)}")
        assert is_eq(res, [
            ("England", 2),
            ("Israel", 2),
            ("United States", 1),
        ])

        res = preql("join(p:Person, c:Country) {country:c.name => citizens: p.name}")
        # TODO Array, not string
        assert is_eq(res, [
            ("England", "Eric Blaire|H.G. Wells"),
            ("Israel", "Erez Shinan|Ephraim Kishon"),
            ("United States", "John Steinbeck"),
        ]), list(res)

        res = preql("join(p:Person, c:Country) {country:c.name => citizens: p.name, count(p.id)}")
        # TODO Array, not string
        assert is_eq(res, [
            ("England", "Eric Blaire|H.G. Wells", 2),
            ("Israel", "Erez Shinan|Ephraim Kishon", 2),
            ("United States", "John Steinbeck", 1),
        ])

        res = preql('[1,2,3]{=>sum(value*value)}')
        assert res == [{'sum': 14}], list(res)

    def _test_table_ops(self, preql):
        # TODO should be consistent - always table, or always array
        res = preql("""[1,2,3]""")
        assert res == [1,2,3], res
        res = preql("""[1,2,3] + [5,6]""")
        assert res == [1,2,3,5,6]

        res = preql("""[1,2,3] | [3,4]""")
        assert res == [1,2,3,4]

        res = preql("""[1,2,3] - [3,4]""")
        assert res == [1,2]

        res = preql("""[1,2,3]{v:value*2}[v < 5]""")
        assert res == [{'v': 2}, {'v': 4}], res

        res = preql("""[1,2,3]{v:value*2}[v in [2,6]]""")
        assert res == [{'v': 2}, {'v': 6}], res

        res = preql("""[1,2,3]{v:value*2}[v ^in [2,6]]""")
        assert res == [{'v': 4}], res

        res = preql("""enum([1,8,4,4])[index==value]{value}""")
        assert res == [{'value': 1}, {'value': 4}]

    def _test_temptable(self, preql):
        english_speakers = [
            ("Eric Blaire", "England"),
            ("H.G. Wells", "England"),
            ("John Steinbeck", "United States"),
        ]

        preql.exec("""english_countries = temptable(Country[language=="en"])""")
        res = preql("english_countries{name}")
        assert is_eq(res, [("England",), ("United States",)])

        preql.exec("""names = temptable(Person{name}) """)
        res = preql('names')
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

        res = preql(""" temptable(temptable(Person)[name=="Erez Shinan"]){name} """) # 2 temp tables
        assert is_eq(res, [("Erez Shinan",)])



    def test_structs(self):
        preql = self.Preql()
        preql.load('box_circle.pql', rel_to=__file__)

        res1 = preql.circles_contained1()
        res2 = preql.circles_contained2()
        res3 = preql("temptable(circles_contained2())")

        assert res1 == res2, (res1, res2)
        assert res2 == res3, (list(res2), list(res3))



    def test_simple1(self):
        # TODO uncomment these tests
        preql = self.Preql()
        preql.load('simple1.pql', rel_to=__file__)

        assert preql.english() == [{'id': 2, 'name': 'Eric Blaire'}, {'id': 3, 'name': 'H.G. Wells'}]
        assert preql.by_country('Israel') == [{'id': 1, 'name': 'Erez Shinan', 'country': 'Israel'}]

        assert preql.english2() == [{'name': 'H.G. Wells'}, {'name': 'Eric Blaire'}]
        # assert preql.english3().json() == [{'n': 'H.G. Wells'}, {'n': 'Eric Blaire'}] # TODO

        # assert preql.person1() == [{'id': 1, 'name': 'Erez Shinan', 'country': 'Israel'}]
        # assert preql.person1b() == [{'id': 2, 'name': 'Eric Blaire', 'country': 'England'}]
        assert preql.demography() == [{'country': 'England', 'population': 2}, {'country': 'Israel', 'population': 1}]

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
        preql.exec('''
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

    @skip("Not ready yet")
    def test_self_reference(self):
        assert False, "Not ready yet"
        preql = self.Preql()
        preql.exec('''
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
        preql.exec('''
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