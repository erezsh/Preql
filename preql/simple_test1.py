
from . import Preql

def main():
    preql = Preql()
    preql.load('simple1.pql', rel_to=__file__)

    # print( preql.english() )
    print( list(preql.english() ) )
    print( list(preql.by_country("England") ) )
    print( list(preql.english2() ) )
    print( list(preql.english3() ) )
    print( list(preql.person1() ) )
    print( list(preql.person1b() ) )
    print( list(preql.person1c() ) )
    print( list(preql.f3() ) )
    print( list(preql.demography() ) )
    print( list(preql.page_test() ) )

main()