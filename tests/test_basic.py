from unittest import TestCase

from preql import Preql
# from .interpreter import Interpreter

class BasicTests(TestCase):

    def test_simple1(self):
        preql = Preql()
        preql.load('simple1.pql', rel_to=__file__)

        assert preql.english().json() == [{'id': 2, 'name': 'Eric Blaire'}, {'id': 3, 'name': 'H.G. Wells'}]
        assert preql.by_country('Israel').json() == [{'id': 1, 'name': 'Erez Shinan', 'country': 'Israel'}]

        assert preql.english2().json() == [{'name': 'H.G. Wells'}, {'name': 'Eric Blaire'}]
        # assert preql.english3().json() == [{'n': 'H.G. Wells'}, {'n': 'Eric Blaire'}] # TODO

        assert preql.person1().json() == [{'id': 1, 'name': 'Erez Shinan', 'country': 'Israel'}]
        assert preql.person1b().json() == [{'id': 2, 'name': 'Eric Blaire', 'country': 'England'}]
        assert preql.demography().json() == [{'country': 'England', 'population': 2}, {'country': 'Israel', 'population': 1}]

        res = [{'country': 'England', 'population': ['Eric Blaire', 'H.G. Wells']}, {'country': 'Israel', 'population': ['Erez Shinan']}]
        assert preql('Person {country => population: name}').json() == res


    def test_simple2(self):
        preql = Preql()
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
               {'name': 'Israel', 'citizens.name': ['Erez Shinan', 'Ephraim Kishon']},
               {'name': 'United States', 'citizens.name': ['John Steinbeck']}]
        assert preql.citizens_list().json() == res

        res = [{'id': 1, 'name': 'Erez Shinan', 'country.name': 'Israel'},
               {'id': 2, 'name': 'Ephraim Kishon', 'country.name': 'Israel'},
               {'id': 3, 'name': 'Eric Blaire', 'country.name': 'England'},
               {'id': 4, 'name': 'H.G. Wells', 'country.name': 'England'},
               {'id': 5, 'name': 'John Steinbeck', 'country.name': 'United States'}]

        assert preql.person_and_country().json() == res
        # print( preql.person_and_country_join().json() )   # TODO
        # print( preql.person_and_country_freejoin().json() )

    def test_m2m(self):
        preql = Preql()
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
        assert (preql('A {a: value, b: ab.b.value}').json()) == res
        assert (preql('B {a: ab.a.value, b: value}').json()) == res

        assert (preql('B [ab.a.value=2] {value}').json()) == [{'value': 4}]

        assert (preql('A_B [a.value=2] {v:b.value}').json()) == [{'v': 4}]

        res = [{'a.value': 0, 'b.value': 0}, {'a.value': 1, 'b.value': 2},
               {'a.value': 2, 'b.value': 4}, {'a.value': 3, 'b.value': 6},
               {'a.value': 4, 'b.value': 8}]
        assert (preql('A_B {a.value, b.value}').json() ) == res
        assert (preql('A_B {a, b} {a.value, b.value}').json() ) == res

    def test_self_reference(self):
        preql = Preql()
        preql.exec('''
            table Person:
                name: string
                parent: Person? -> children
        ''')

        abraham = preql('new Person("Abraham", null)').row_id
        isaac = preql('new Person("Isaac", ab)', ab=abraham).row_id
        jacob = preql('new Person("Jacob", isaac)', isaac=isaac).row_id
        esau = preql('new Person("Esau", isaac)', isaac=isaac).row_id

        assert (preql('Person[name="Jacob"] {name: parent.name}').json()) == [{'name': 'Isaac'}]
        # assert (preql('Person[name="Jacob"] {name: parent.parent.name}').json()) == [{'name': 'Abraham'}] # TODO

        res = [{'name': 'Abraham', 'count_children': 1}, {'name': 'Isaac', 'count_children': 2}]
        assert ( preql('Person {name => count(children)}').json()) == res

        res = [{'name': 'Abraham', 'children.name': ['Isaac']}, {'name': 'Isaac', 'children.name': ['Jacob', 'Esau']}]
        assert ( preql('Person {name => children.name}').json()) == res

        # assert ( preql('Person {name => children.name}').json()) == res

    def test_m2m_with_self_reference(self):
        preql = Preql()
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

        a1 = preql('new A("a1", null)').row_id
        a2 = preql('new A("a2", a1)', a1=a1).row_id
        a3 = preql('new A("a3", a1)', a1=a1).row_id

        b1 = preql('new B("b1", null)').row_id
        b2 = preql('new B("b2", null)').row_id

        preql('new A_B(a1, b1)', a1=a1, b1=b1).row_id
        preql('new A_B(a2, b2)', a2=a2, b2=b2).row_id

        # TODO Table identity is messed up!
        # The rules should be as followed (at least for now):
        # * Reference always brings a new table ( so join(A,A) will joins two separate selects )
        # * get_attr should returns a new table, but cached, according to the originating table
        #   so {ab} creates a new reference, and {children.ab} creates yet another new reference
        #   so that two separate joins occur!
        res = [{'ab.b.name': 'b1', 'children.ab.b.name': 'b2'}]
        assert( preql('A {ab.b.name, children.ab.b.name}').json() ) == res

        assert( preql('A[ab.b.name="b1", children.ab.b.name="b2"] {name}').json() ) == [{'name': 'a1'}]
        # print( preql('A {children.ab.b.name}').json() )
        # print( preql('A[ab.b.name="b1"] {children}').json() ) # TODO get it to work