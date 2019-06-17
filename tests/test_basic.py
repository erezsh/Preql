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