"""
  Benchmarking Queries
========================

This module compares the performance of hand-written SQL
against the SQL produced by Preql.

Tested on the Chinook database. See examples/chinook.pql


  Results on SQLite (1000 iterations)
---------------------------------------

test1 - simple selection and projection
    * SQL  : 0.010223775863647462
    * Preql: 0.010765978813171387
test2 - multijoin and groupby
    * SQL  : 0.03636674237251282
    * Preql: 0.03679810237884522

"""

import time

from preql import Preql, T

p = Preql()
p.load('../examples/chinook.pql', rel_to=__file__)

ITERS = 100

def _measure_sql(sql, iters=ITERS):
    start = time.time()
    for i in range(iters):
        l = p._interp.state.db._backend_execute_sql(sql).fetchall()
    elapsed = time.time() - start
    return l, elapsed / iters

def test1():
    print("test1 - simple selection and projection")
    sql = r"select *, a.Title from tracks t join albums a using(AlbumId) where Title like '%r%'"
    l1, elapsed = _measure_sql(sql)
    print('SQL:', elapsed)

    join = r'join(t: tracks, a: albums) {..., a.Title} [Title like "%r%"]'
    sql = p(f'inspect_sql({join})')
    l2, elapsed = _measure_sql(sql)
    print('Preql:', elapsed)

    assert len(l1) == len(l2)


def test2():
    print("test2 - multijoin and groupby")
    sql = r"""
    SELECT
        t.TrackId AS [TrackId],
        t.Name AS [Name],
        t.MediaTypeId AS [MediaTypeId],
        t.Composer AS [Composer],
        t.Milliseconds AS [Milliseconds],
        t.Bytes AS [Bytes],
        t.UnitPrice AS [UnitPrice],
        alb.Title AS [Album],
        art.Name AS [Artist],
        g.Name AS [Genre],
        group_concat(p.Name, "||") AS [Categories_item]
    FROM artists art
    JOIN albums alb
    JOIN tracks t
    JOIN genres g
    JOIN playlist_track pt
    JOIN playlists p
        ON (art.ArtistId = alb.ArtistId)
        AND (alb.AlbumId = t.AlbumId)
        AND (t.GenreId = g.GenreId)
        AND (t.TrackId = pt.TrackId)
        AND (pt.PlaylistId = p.PlaylistId)
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    """
    l1, elapsed = _measure_sql(sql)
    print('SQL:', elapsed)

    join = """
        join(
            art: artists,
            alb: albums,
            t: tracks,
            g: genres,
            pt: playlist_track,
            p: playlists
        ) {
            ...t !GenreId !AlbumId
            Album: alb.Title
            Artist: art.Name
            Genre: g.Name
                =>
            Categories: p.Name
        }
    """
    sql = p(f'inspect_sql({join})')
    l2, elapsed = _measure_sql(sql)
    print('Preql:', elapsed)

    assert len(l1) == len(l2)


print(f"Testing with {ITERS} iterations")
test1()
test2()