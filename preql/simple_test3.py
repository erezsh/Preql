
from . import Preql

def main():
    preql = Preql()
    preql.load('simple3.pql', rel_to=__file__)

    print( list(preql('Person {name, city.country.name}' ) ) )
    print( list(preql('Person {city.country.language => name}' ) ) )
    print( list(preql('Country {name => cities.citizens.name}' ) ) )

    # print( list(preql('City {country.name => citizens}' ) ) )

    preql.start_repl()

main()