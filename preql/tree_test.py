
from . import Preql

def main():
    preql = Preql()
    preql.load('tree.pql', rel_to=__file__)

    print( list(preql('Tree' ) ) )
    print( list(preql('Tree {value, parent.value}' ) ) )

    print( list(preql('Tree {value, parent.parent.value}' ) ) ) # Not working yet
    # print( list(preql('Person {name, city.country.name}' ) ) )

    # print( list(preql('City {country.name => citizens}' ) ) )

    preql.start_repl()

main()