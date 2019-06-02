
from . import Preql

def main():
    preql = Preql()
    preql.load('simple2.pql', rel_to=__file__)

    print( list(preql.explicit_join() ) )
    print( list(preql('explicit_join()' ) ) )
    print( list(preql('explicit_join() {c.name, p.name}' ) ) )
    print( list(preql('explicit_join() {c}' ) ) )
    print()
    print( list(preql('english_speakers()' ) ) )
    print( list(preql('person_and_language()' ) ) )
    print( list(preql('country_by_language()' ) ) )

    print( list(preql('population_count()' ) ) )

    print( list(preql('Person {name, country.name}' ) ) )

    # # print( list(preql('Country {name, citizens}' ) ) )     #XXX SHOULD NOT BE ALLOWED, due to types
    print( list(preql('Country {name => citizens}' ) ) )
    print( list(preql('Country {name => count(citizens)}' ) ) )
    # # print( list(preql('Country {name, count(citizens)}' ) ) )  # XXX SHOULD NOT BE ALLOWED, due to types

    print( list(preql('Person {country.name => name} ' ) ) )
    print( list(preql('Country {name => citizens.name}' ) ) )

    # print( list(preql('Person {country {name}}' ) ) )

    preql.start_repl()
    # print( list(preql.use_join() ) )

main()