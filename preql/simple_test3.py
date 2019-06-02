
from . import Preql

def main():
    preql = Preql()
    preql.load('simple3.pql', rel_to=__file__)

    print( list(preql('Person {name, city.country.name}' ) ) )
    print( list(preql('Person {city.country.language => name}' ) ) )

    print( list(preql('City {country.name => citizens}' ) ) )

    # print( list(preql('Person {country {name}}' ) ) )

    preql.start_repl()
    # print( list(preql.use_join() ) )

main()