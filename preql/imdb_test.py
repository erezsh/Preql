
from . import Preql
# from .interpreter import Interpreter


schema = """
table actors:
    first_name: string
    last_name: string
    gender: string 

table directors:
    first_name: string
    last_name: string

table movies:
    name: string
    year: integer
    rank: float

table movies_genres:
    movie_id: movies -> genres
    genre: string

table roles:
    actor_id: actors -> roles
    movie_id: movies -> roles
    role: string

table directors_genres:
  director_id: directors -> genres
  genre: string
  prob: float

table movies_directors:
  director_id: directors
  movie_id: movies


# count_by_year(T) = T {year => count(id)}

"""


def main():
    preql = Preql('imdb.db')
    preql.exec(schema)

    # print(i.call_func('count_best_movies_by_year', []))

    # print(i.eval_expr('movies [rank > 9] {year => count(id)}'))

    # print(i.eval_expr('movies [rank > 9] :limit(5) {name, genres}'))   # TODO: What do to? Aggregate? Error?

    # print(i.eval_expr('movies [rank > 9] :limit(5) {name => genres}'))    # TODO throw proper error, as id doesn't exist

    # print(i.eval_expr('movies [rank > 9] :limit(5) {name => genres.genre}'))

    # print( preql( 'count(movies)' ) )
    # print( preql( 'count(movies [rank>9])' ) )

    # print( preql( 'movies :count()' ) )
    # print( preql( 'movies [rank>9] :count()' ) )

    # print( preql( 'movies {round(rank) => count(id)}' ) )

    # print( preql( 'movies[ rank > 9 ].limit(5)' ) )
    # return
    print( preql( 'movies.order(-name).count()' ) )

    print( preql( 'movies' ) )
    print( preql( 'movies.count()' ) )
    print( preql( 'movies[ rank > 9 ].count()' ) )
    print( preql( 'movies[ rank > 9 ]' ) )

    print( preql( 'movies {name}' ) )
    print( preql( 'movies {n:name, r:rank}' ) )
    print( preql( 'movies [rank>9] {n:name, r:rank}' ) )
    print( preql( 'movies {year => count(id)}' ) )
    print( preql( 'movies {round(rank) => count(id)}' ) )
    print( preql( 'movies {n:name, r:rank} [r>9].count()' ) )

    ### MORE TESTS
    # movies[rank >= 9.9, year>2000]  


    # print( preql( 'movies {r: round(rank) => c: count(id)}[ r > 9] ' ) )

    

    # print( preql( 'movies[id=0]' ) )
    # print( preql( 'movies(1)' ) )

    preql.start_repl()

    # print(i.eval_query('count_by_year( movies [rank > 9] )'))     # TODO: Not implemented yet

main()